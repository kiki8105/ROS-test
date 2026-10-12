import math
import unittest

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from my_turtle_pkg.pure_pursuit_obstacle_v2 import (
    astar_path,
    grid_to_world,
    GridSpec,
    nearest_free_cell,
    occupancy_grid_to_blocked,
    propagate_odom_pose,
    pure_pursuit_arc_is_free,
    PurePursuitObstacleV2,
    validate_mission_waypoints,
    WAFFLE_PI_FRONT_X,
    WAFFLE_PI_LIDAR_X,
    world_to_grid,
)
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan


def make_map(width=40, height=40, resolution=0.1, origin=(-1.0, -1.0)):
    message = OccupancyGrid()
    message.header.frame_id = 'map'
    message.info.width = width
    message.info.height = height
    message.info.resolution = resolution
    message.info.origin.position.x = float(origin[0])
    message.info.origin.position.y = float(origin[1])
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (width * height)
    return message


def make_pose(x=0.0, y=0.0, yaw=0.0, frame='map'):
    message = PoseWithCovarianceStamped()
    message.header.frame_id = frame
    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def make_goal(x=1.5, y=0.0, yaw=0.0, frame='map'):
    message = PoseStamped()
    message.header.frame_id = frame
    message.pose.position.x = float(x)
    message.pose.position.y = float(y)
    message.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def make_odom(x=0.0, y=0.0, yaw=0.0):
    message = Odometry()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_footprint'
    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def make_scan(default=math.inf):
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = math.pi / 180.0
    scan.angle_max = math.pi
    scan.range_min = 0.12
    scan.range_max = 10.0
    scan.ranges = [default] * 361
    return scan


def paint_sector(scan, centre_degrees, half_width_degrees, distance):
    centre = math.radians(centre_degrees)
    half_width = math.radians(half_width_degrees)
    for index in range(len(scan.ranges)):
        angle = scan.angle_min + index * scan.angle_increment
        error = (angle - centre + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error) <= half_width:
            scan.ranges[index] = float(distance)


def scan_with_front_gap(gap, left=2.0, right=0.55):
    scan = make_scan()
    raw_range = gap + WAFFLE_PI_FRONT_X - WAFFLE_PI_LIDAR_X
    paint_sector(scan, 0.0, 7.0, raw_range)
    paint_sector(scan, 45.0, 12.0, left)
    paint_sector(scan, 90.0, 20.0, left)
    paint_sector(scan, -45.0, 12.0, right)
    paint_sector(scan, -90.0, 20.0, right)
    return scan


class CaptureTwistPublisher:

    def __init__(self):
        self.commands = []

    def publish(self, message):
        self.commands.append((message.linear.x, message.angular.z))


class CapturePathPublisher:

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class GridAndAStarTests(unittest.TestCase):

    def test_code_mission_waypoints_require_finite_xy_pairs(self):
        self.assertEqual(
            validate_mission_waypoints([(1, 2.5), [-3.0, 4]]),
            [(1.0, 2.5), (-3.0, 4.0)],
        )
        with self.assertRaises(ValueError):
            validate_mission_waypoints([(1.0, 2.0, 3.0)])
        with self.assertRaises(ValueError):
            validate_mission_waypoints([(math.nan, 0.0)])

    def test_occupancy_threshold_and_unknown_policy(self):
        message = make_map(width=3, height=2, resolution=1.0, origin=(0, 0))
        message.data = [0, 100, -1, 49, 50, 0]

        blocked, _ = occupancy_grid_to_blocked(
            message, occupied_threshold=50,
            unknown_is_blocked=True, inflation_radius=0.0)
        self.assertEqual(blocked, [False, True, True, False, True, False])

        blocked, _ = occupancy_grid_to_blocked(
            message, occupied_threshold=50,
            unknown_is_blocked=False, inflation_radius=0.0)
        self.assertEqual(blocked, [False, True, False, False, True, False])

    def test_world_grid_conversion_supports_rotated_origin(self):
        spec = GridSpec(10, 10, 0.5, 2.0, -1.0, math.pi / 2.0)
        world = grid_to_world(3, 4, spec)
        self.assertEqual(world_to_grid(*world, spec), (3, 4))

    def test_astar_refuses_diagonal_corner_cut(self):
        blocked = [False, True, True, False]
        self.assertIsNone(astar_path(blocked, 2, 2, (0, 0), (1, 1)))

    def test_astar_uses_open_diagonal_shortest_path(self):
        path = astar_path([False] * 25, 5, 5, (0, 0), (4, 4))
        self.assertEqual(path, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

    def test_nearest_free_radius_is_metric_not_ceil_cell(self):
        spec = GridSpec(3, 3, 1.0, 0.0, 0.0, 0.0)
        blocked = [False] * 9
        blocked[4] = True
        self.assertIsNone(nearest_free_cell(blocked, spec, (1, 1), 0.01))
        self.assertIn(
            nearest_free_cell(blocked, spec, (1, 1), 1.0),
            {(0, 1), (1, 0), (1, 2), (2, 1)},
        )

    def test_odom_delta_is_rotated_into_map_frame(self):
        result = propagate_odom_pose(
            map_anchor=(2.0, 3.0, math.pi / 2.0),
            odom_anchor=(10.0, 5.0, 0.0),
            odom_current=(11.0, 5.0, 0.2),
        )
        self.assertAlmostEqual(result[0], 2.0, places=6)
        self.assertAlmostEqual(result[1], 4.0, places=6)
        self.assertAlmostEqual(result[2], math.pi / 2.0 + 0.2, places=6)

    def test_predicted_pure_pursuit_arc_detects_blocked_cell(self):
        spec = GridSpec(12, 12, 1.0, -5.0, -5.0, 0.0)
        blocked = [False] * (spec.width * spec.height)
        self.assertTrue(pure_pursuit_arc_is_free(
            blocked, spec, 0.0, 0.0, 0.0, 2.0, math.pi / 4.0))

        # The midpoint of this quarter-circle is near world (1.0, 0.41).
        blocked[5 * spec.width + 6] = True
        self.assertFalse(pure_pursuit_arc_is_free(
            blocked, spec, 0.0, 0.0, 0.0, 2.0, math.pi / 4.0))


class IntegratedNodeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = PurePursuitObstacleV2()
        self.node.timer.cancel()
        self.twists = CaptureTwistPublisher()
        self.paths = CapturePathPublisher()
        self.node.publisher_ = self.twists
        self.node.path_publisher = self.paths

    def tearDown(self):
        self.node.destroy_node()

    def tick(self):
        self.node.last_control_time = (
            self.node.get_clock().now() - Duration(seconds=0.1))
        self.node.control_loop()
        return self.twists.commands[-1]

    def set_enabled(self, enabled):
        result = self.node.set_parameters([
            Parameter('drive_enabled', Parameter.Type.BOOL, enabled),
        ])[0]
        self.assertTrue(result.successful, result.reason)

    def prepare_plan(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())
        self.node.goal_callback(make_goal())
        self.tick()
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertGreaterEqual(len(self.node.path), 2)

    def test_disabled_node_plans_and_publishes_path_but_stays_stopped(self):
        self.prepare_plan()

        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))
        self.assertGreater(len(self.paths.messages[-1].poses), 1)
        self.assertEqual(self.paths.messages[-1].header.frame_id, 'map')
        self.assertAlmostEqual(self.node.path[0][0], 0.0)
        self.assertAlmostEqual(self.node.path[-1][0], 1.5)

    def test_clear_scan_and_enable_publish_forward_cmd_vel(self):
        self.prepare_plan()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)

        linear, angular = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0, delta=0.02)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_front_obstacle_enters_turn_and_latches_direction(self):
        self.prepare_plan()
        self.set_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=2.0, right=0.55))

        linear, angular = self.tick()
        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)

        self.node.scan_callback(
            scan_with_front_gap(0.30, left=0.55, right=2.0))
        _, angular = self.tick()
        self.assertGreater(angular, 0.0)
        self.assertEqual(self.node.turn_direction, 1.0)

    def test_replanning_is_suppressed_during_turn(self):
        self.prepare_plan()
        self.set_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=2.0, right=0.55))
        self.tick()
        plan_count = self.node.plan_count

        self.node.current_y = 1.0
        self.node.pose_callback(make_pose(0.0, 1.0, 0.0))
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=0.55, right=2.0))
        self.tick()

        self.assertEqual(self.node.plan_count, plan_count)
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)

        result = self.node.set_parameters([
            Parameter('inflation_radius', Parameter.Type.DOUBLE, 0.30),
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=0.55, right=2.0))
        self.tick()
        self.assertEqual(self.node.plan_count, plan_count)
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)
        self.assertTrue(self.node.costmap_dirty)

    def test_failed_new_goal_clears_old_path_and_stops(self):
        self.prepare_plan()
        self.assertTrue(self.node.path)

        self.node.goal_callback(make_goal(50.0, 50.0))
        self.tick()

        self.assertFalse(self.node.path)
        self.assertIsNone(self.node.active_goal)
        self.assertEqual(self.node.planner_state, 'NO_PATH')
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))
        self.assertEqual(len(self.paths.messages[-1].poses), 0)

    def test_failed_deviation_replan_cannot_overwrite_stop(self):
        self.prepare_plan()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)

        # Insert a full-height wall after the initial plan, then move the
        # reported pose far enough away from that plan to force replanning.
        wall_x = 17
        for grid_y in range(self.node.grid_spec.height):
            index = grid_y * self.node.grid_spec.width + wall_x
            self.node.blocked[index] = True
        self.node.pose_callback(make_pose(0.0, 0.6))
        self.node.last_plan_time = None

        command = self.tick()

        self.assertEqual(self.node.planner_state, 'NO_PATH')
        self.assertFalse(self.node.path)
        self.assertIsNone(self.node.active_goal)
        self.assertEqual(command, (0.0, 0.0))
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_planning_parameter_change_rebuilds_raw_map_and_replans(self):
        message = make_map()
        message.data[35 * message.info.width + 35] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose())
        self.node.goal_callback(make_goal())
        self.tick()
        old_count = sum(self.node.blocked)
        old_plan_count = self.node.plan_count

        result = self.node.set_parameters([
            Parameter('inflation_radius', Parameter.Type.DOUBLE, 0.30),
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.tick()

        self.assertGreater(sum(self.node.blocked), old_count)
        self.assertGreater(self.node.plan_count, old_plan_count)

    def test_stale_scan_stops_after_motion(self):
        self.prepare_plan()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.assertGreater(self.tick()[0], 0.0)

        self.node.last_scan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'WAIT_SCAN')

    def test_odom_heartbeat_keeps_pose_fresh_between_amcl_updates(self):
        self.node.map_callback(make_map())
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.pose_callback(make_pose(0.0, 0.0, 0.0))
        self.node.goal_callback(make_goal())
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)

        self.node.last_pose_time = (
            self.node.get_clock().now() - Duration(seconds=5.0))
        self.node.odom_callback(make_odom(0.01, 0.0, 0.0))
        linear, _ = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(self.node.current_x, 0.01, places=4)

    def test_blocked_start_is_rejected_instead_of_snapped_across_wall(self):
        message = make_map()
        start_cell = (10, 10)
        message.data[start_cell[1] * message.info.width + start_cell[0]] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose(0.0, 0.0))
        self.node.goal_callback(make_goal())

        self.tick()

        self.assertFalse(self.node.path)
        self.assertEqual(self.node.planner_state, 'NO_PATH')
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_large_heading_error_near_side_obstacle_blocks_spin(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.0, math.pi))
        self.node.goal_callback(make_goal())
        self.tick()
        scan = make_scan()
        paint_sector(scan, 90.0, 5.0, 0.16)
        self.node.scan_callback(scan)
        self.set_enabled(True)

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_non_map_pose_is_rejected_before_map_arrives(self):
        self.node.pose_callback(make_pose(frame='odom'))

        self.assertFalse(self.node.is_localized)
        self.assertIsNone(self.node.last_pose_time)

    def test_code_mission_advances_each_wp_and_latches_completion(self):
        self.node.mission_waypoints = [(0.5, 0.0), (1.0, 0.0)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())
        self.tick()

        self.assertEqual(self.node.mission_waypoint_index, 0)
        self.assertEqual(self.node.requested_goal, (0.5, 0.0))
        self.assertEqual(self.node.goal_source, 'MISSION')
        self.assertEqual(self.node.planner_state, 'READY')

        self.set_enabled(True)
        self.node.pose_callback(make_pose(0.5, 0.0))
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.mission_waypoint_index, 1)
        self.assertEqual(self.node.requested_goal, (1.0, 0.0))
        self.assertTrue(self.node.replan_requested)
        self.assertFalse(self.node.path)

        # The next control tick plans the second A* leg but remains stopped
        # because no scan is supplied in this unit test.
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.planner_state, 'READY')
        self.node.pose_callback(make_pose(1.0, 0.0))
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.completed)
        self.assertTrue(self.node.mission_completed)
        self.assertFalse(self.node.mission_active)
        self.assertEqual(self.node.planner_state, 'MISSION_COMPLETED')
        self.assertIsNone(self.node.requested_goal)

        # Map changes and localization drift cannot restart a finished mission.
        previous_plan_count = self.node.plan_count
        changed_map = make_map()
        changed_map.data[35 * changed_map.info.width + 35] = 100
        self.node.map_callback(changed_map)
        self.node.pose_callback(make_pose(0.5, 0.0))
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.plan_count, previous_plan_count)
        self.assertEqual(self.node.planner_state, 'MISSION_COMPLETED')

    def test_external_goal_permanently_cancels_code_mission(self):
        self.node.mission_waypoints = [(0.5, 0.0), (1.0, 0.0)]
        self.assertTrue(self.node.start_code_mission())

        self.node.goal_callback(make_goal(1.5, 0.5))

        self.assertFalse(self.node.mission_active)
        self.assertFalse(self.node.mission_completed)
        self.assertEqual(self.node.mission_waypoint_index, -1)
        self.assertEqual(self.node.goal_source, 'EXTERNAL')
        self.assertEqual(self.node.requested_goal, (1.5, 0.5))

    def test_unreachable_code_wp_stops_mission_without_skipping(self):
        self.node.mission_waypoints = [(50.0, 50.0), (0.5, 0.0)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())

        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertFalse(self.node.mission_active)
        self.assertTrue(self.node.mission_failed)
        self.assertEqual(self.node.mission_waypoint_index, 0)
        self.assertEqual(self.node.planner_state, 'MISSION_FAILED')
        self.assertIsNone(self.node.requested_goal)
        self.assertFalse(self.node.path)

        plan_count = self.node.plan_count
        changed_map = make_map()
        changed_map.data[35 * changed_map.info.width + 35] = 100
        self.node.map_callback(changed_map)
        self.tick()
        self.assertEqual(self.node.plan_count, plan_count)
        self.assertEqual(self.node.planner_state, 'MISSION_FAILED')


if __name__ == '__main__':
    unittest.main()
