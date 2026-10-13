import math
import unittest

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan

from my_turtle_pkg.pure_pursuit_v3 import (
    PurePursuitV3,
    WAFFLE_PI_FRONT_X,
    WAFFLE_PI_LIDAR_X,
    choose_turn_direction,
    front_body_clearance,
    propagate_odom_pose,
    scan_sector_distance,
)


def make_scan(default=math.inf):
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = math.pi / 180.0
    scan.range_min = 0.12
    scan.range_max = 10.0
    scan.ranges = [default] * 361
    return scan


def paint_sector(scan, centre_degrees, half_width_degrees, distance):
    for index in range(len(scan.ranges)):
        angle = math.degrees(
            scan.angle_min + index * scan.angle_increment)
        error = (angle - centre_degrees + 180.0) % 360.0 - 180.0
        if abs(error) <= half_width_degrees:
            scan.ranges[index] = distance


def scan_with_front_gap(
        gap, left=2.0, right=0.50, front_half_width_degrees=8.0):
    scan = make_scan()
    # 정면 광선에서 body gap = lidar_x + range - body_front_x.
    raw_range = gap + WAFFLE_PI_FRONT_X - WAFFLE_PI_LIDAR_X
    paint_sector(
        scan, 0.0, front_half_width_degrees, raw_range)
    paint_sector(scan, 45.0, 18.0, left)
    paint_sector(scan, 90.0, 18.0, left)
    paint_sector(scan, -45.0, 18.0, right)
    paint_sector(scan, -90.0, 18.0, right)
    return scan


def make_odom(x, y, yaw, stamp_seconds=None):
    odom = Odometry()
    if stamp_seconds is not None:
        seconds = int(stamp_seconds)
        odom.header.stamp.sec = seconds
        odom.header.stamp.nanosec = int(
            round((float(stamp_seconds) - seconds) * 1e9))
    odom.pose.pose.position.x = float(x)
    odom.pose.pose.position.y = float(y)
    odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return odom


def make_map_pose(x, y, yaw, stamp_seconds=None):
    pose = PoseWithCovarianceStamped()
    if stamp_seconds is not None:
        seconds = int(stamp_seconds)
        pose.header.stamp.sec = seconds
        pose.header.stamp.nanosec = int(
            round((float(stamp_seconds) - seconds) * 1e9))
    pose.pose.pose.position.x = float(x)
    pose.pose.pose.position.y = float(y)
    pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


class CapturePublisher:
    def __init__(self):
        self.commands = []

    def publish(self, message):
        self.commands.append((message.linear.x, message.angular.z))


class ScanGeometryTests(unittest.TestCase):
    def test_odom_delta_is_rotated_into_map_frame(self):
        x, y, yaw = propagate_odom_pose(
            map_anchor=(5.0, 10.0, math.pi / 2.0),
            odom_anchor=(2.0, 3.0, 0.0),
            odom_current=(3.0, 3.0, math.pi / 4.0),
        )
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 11.0)
        self.assertAlmostEqual(yaw, 3.0 * math.pi / 4.0)

    def test_odom_yaw_wraps_across_signed_pi(self):
        _, _, yaw = propagate_odom_pose(
            map_anchor=(0.0, 0.0, math.radians(170.0)),
            odom_anchor=(0.0, 0.0, math.radians(-170.0)),
            odom_current=(0.0, 0.0, math.radians(-150.0)),
        )
        self.assertAlmostEqual(yaw, math.radians(-170.0))

    def test_front_clearance_uses_body_frame_and_rejects_one_ray(self):
        scan = make_scan()
        centre = len(scan.ranges) // 2
        raw_range = 0.30 + WAFFLE_PI_FRONT_X - WAFFLE_PI_LIDAR_X

        scan.ranges[centre] = raw_range
        self.assertTrue(math.isinf(
            front_body_clearance(scan, half_width=0.18, min_hits=3)))

        for index in range(centre - 1, centre + 2):
            scan.ranges[index] = raw_range
        self.assertAlmostEqual(
            front_body_clearance(scan, half_width=0.18, min_hits=3),
            0.30,
            delta=0.001)

    def test_door_posts_outside_body_corridor_do_not_block(self):
        scan = make_scan()
        # base_footprint (x=0.35, y=+/-0.25)에 있는 두 문틀을 그린다.
        for y in (-0.25, 0.25):
            relative_x = 0.35 - WAFFLE_PI_LIDAR_X
            angle = math.atan2(y, relative_x)
            distance = math.hypot(relative_x, y)
            centre = round((angle - scan.angle_min) / scan.angle_increment)
            for index in range(centre - 2, centre + 3):
                scan.ranges[index] = distance

        self.assertTrue(math.isinf(
            front_body_clearance(scan, half_width=0.18, min_hits=3)))

    def test_sector_and_turn_direction_helpers(self):
        scan = make_scan()
        paint_sector(scan, 0.0, 10.0, 0.70)
        self.assertAlmostEqual(
            scan_sector_distance(
                scan, 0.0, math.radians(15), percentile=0.05),
            0.70)
        self.assertEqual(choose_turn_direction(2.0, 0.5), 1.0)
        self.assertEqual(choose_turn_direction(0.5, 2.0), -1.0)
        self.assertEqual(choose_turn_direction(1.0, 1.02, previous=-1.0), -1.0)


class PurePursuitV3NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = PurePursuitV3()
        self.node.timer.cancel()
        self.capture = CapturePublisher()
        self.node.publisher = self.capture

        pose = PoseWithCovarianceStamped()
        target_yaw = math.atan2(0.56, 6.27)
        pose.pose.pose.orientation.z = math.sin(target_yaw / 2.0)
        pose.pose.pose.orientation.w = math.cos(target_yaw / 2.0)
        self.node.pose_callback(pose)

    def tearDown(self):
        self.node.destroy_node()

    def set_drive_enabled(self, enabled):
        result = self.node.set_parameters([
            Parameter(
                'drive_enabled',
                Parameter.Type.BOOL,
                enabled,
            )
        ])[0]
        self.assertTrue(result.successful, result.reason)

    def tick(self):
        self.node.last_control_time = (
            self.node.get_clock().now() - Duration(seconds=0.1))
        self.node.control_loop()
        return self.capture.commands[-1]

    def test_clear_scan_tracks_goal(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())

        linear, angular = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0, delta=0.02)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_goal_pose_replaces_yaml_path_and_resets_completion(self):
        self.node.current_waypoint_index = len(self.node.path)
        self.node.completed = True
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.25
        goal.pose.position.y = -0.75

        self.node.goal_callback(goal)

        self.assertEqual(self.node.path, [(1.25, -0.75)])
        self.assertEqual(self.node.current_waypoint_index, 0)
        self.assertFalse(self.node.completed)
        self.assertEqual(self.node.goal_subscription.topic_name, '/goal_pose')
        self.assertEqual(
            self.node.get_parameter('global_frame').value,
            'map',
        )

    def test_goal_pose_rejects_wrong_or_empty_frame(self):
        original_path = list(self.node.path)

        wrong_frame = PoseStamped()
        wrong_frame.header.frame_id = 'odom'
        wrong_frame.pose.position.x = 1.25
        wrong_frame.pose.position.y = -0.75
        self.node.goal_callback(wrong_frame)
        self.assertEqual(self.node.path, original_path)

        empty_frame = PoseStamped()
        empty_frame.pose.position.x = 1.25
        empty_frame.pose.position.y = -0.75
        self.node.goal_callback(empty_frame)
        self.assertEqual(self.node.path, original_path)

    def test_goal_pose_is_rejected_while_drive_is_enabled(self):
        original_path = list(self.node.path)
        self.set_drive_enabled(True)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.25
        goal.pose.position.y = -0.75

        self.node.goal_callback(goal)

        self.assertEqual(self.node.path, original_path)

    def test_front_obstacle_stops_and_turns_toward_open_side(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.28, left=2.0, right=0.50))

        linear, angular = self.tick()

        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TURN')

    def test_stale_scan_hard_stops_both_axes(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.28, left=2.0, right=0.50))
        _, turning = self.tick()
        self.assertGreater(turning, 0.0)

        self.node.last_scan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.assertEqual(self.tick(), (0.0, 0.0))

    def test_emergency_obstacle_inside_spin_radius_blocks(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.10, left=2.0, right=2.0))

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_all_nan_scan_is_invalid_and_stops(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan(default=math.nan))

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'WAIT_SCAN')

    def test_drive_gate_stops_and_resets_avoidance(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())
        self.assertGreater(self.tick()[0], 0.0)

        self.set_drive_enabled(False)
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_odom_propagates_pose_and_new_amcl_reanchors(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.odom_callback(make_odom(0.1, 0.0, 0.0))
        initial_yaw = math.atan2(0.56, 6.27)
        self.assertAlmostEqual(
            self.node.current_x, 0.1 * math.cos(initial_yaw), places=5)
        self.assertAlmostEqual(
            self.node.current_y, 0.1 * math.sin(initial_yaw), places=5)

        self.node.pose_callback(
            make_map_pose(0.8, 0.2, math.pi / 2.0))
        self.assertAlmostEqual(self.node.current_x, 0.8)
        self.assertAlmostEqual(self.node.current_y, 0.2)
        self.node.odom_callback(make_odom(0.2, 0.0, 0.0))
        self.assertAlmostEqual(self.node.current_x, 0.8, places=5)
        self.assertAlmostEqual(self.node.current_y, 0.3, places=5)
        self.assertAlmostEqual(
            self.node.current_yaw, math.pi / 2.0, places=5)

    def test_odom_before_amcl_does_not_localize(self):
        self.node.is_localized = False
        self.node.map_anchor_pose = None
        self.node.last_pose_time = None

        self.node.odom_callback(make_odom(1.0, 2.0, 0.3))

        self.assertFalse(self.node.is_localized)
        self.assertIsNone(self.node.last_pose_time)

    def test_first_odom_after_amcl_only_sets_anchor(self):
        before = (
            self.node.current_x,
            self.node.current_y,
            self.node.current_yaw,
        )
        self.node.odom_callback(make_odom(5.0, -3.0, 1.2))

        self.assertEqual(
            (self.node.current_x, self.node.current_y, self.node.current_yaw),
            before)

    def test_invalid_odom_does_not_refresh_pose(self):
        before_pose = (
            self.node.current_x,
            self.node.current_y,
            self.node.current_yaw,
        )
        before_time = self.node.last_pose_time

        self.node.odom_callback(make_odom(math.nan, 0.0, 0.0))

        self.assertEqual(
            (self.node.current_x, self.node.current_y, self.node.current_yaw),
            before_pose)
        self.assertEqual(self.node.last_pose_time, before_time)

    def test_fresh_odom_prevents_sparse_amcl_timeout(self):
        result = self.node.set_parameters([
            Parameter('pose_timeout', Parameter.Type.DOUBLE, 1.0)
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.scan_callback(make_scan())
        self.set_drive_enabled(True)
        self.assertGreater(self.tick()[0], 0.0)

        self.node.last_pose_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.node.odom_callback(make_odom(0.05, 0.0, 0.0))
        self.assertEqual(
            self.node.last_pose_time, self.node.last_odom_time)
        self.assertGreater(self.tick()[0], 0.0)

    def test_stale_amcl_and_odom_stop(self):
        result = self.node.set_parameters([
            Parameter('pose_timeout', Parameter.Type.DOUBLE, 1.0)
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.scan_callback(make_scan())
        self.set_drive_enabled(True)
        self.assertGreater(self.tick()[0], 0.0)
        self.node.last_pose_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))

        self.assertEqual(self.tick(), (0.0, 0.0))

    def test_reenable_does_not_make_stale_pose_fresh(self):
        result = self.node.set_parameters([
            Parameter('pose_timeout', Parameter.Type.DOUBLE, 1.0)
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.scan_callback(make_scan())
        self.set_drive_enabled(True)
        self.assertGreater(self.tick()[0], 0.0)
        self.set_drive_enabled(False)
        self.assertEqual(self.tick(), (0.0, 0.0))

        stale_pose_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.node.last_pose_time = stale_pose_time
        self.set_drive_enabled(True)

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.last_pose_time, stale_pose_time)

    def test_stale_odom_is_not_reused_for_new_amcl_anchor(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.odom_callback(make_odom(0.1, 0.0, 0.0))
        self.node.last_odom_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))

        self.node.pose_callback(make_map_pose(10.0, 3.0, 0.0))
        self.assertIsNone(self.node.odom_anchor_pose)
        self.node.odom_callback(make_odom(0.2, 0.0, 0.0))

        self.assertAlmostEqual(self.node.current_x, 10.0)
        self.assertAlmostEqual(self.node.current_y, 3.0)

    def test_amcl_timestamp_uses_matching_buffered_odom_sample(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 10.0))
        self.node.odom_callback(make_odom(0.1, 0.0, 0.0, 11.0))

        self.node.pose_callback(make_map_pose(5.0, 2.0, 0.0, 10.0))

        self.assertAlmostEqual(self.node.current_x, 5.1)
        self.assertAlmostEqual(self.node.current_y, 2.0)

    def test_stamped_amcl_without_matching_odom_waits_for_new_anchor(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 10.0))

        self.node.pose_callback(make_map_pose(5.0, 2.0, 0.0, 1.0))

        self.assertIsNone(self.node.odom_anchor_pose)
        self.assertAlmostEqual(self.node.current_x, 5.0)
        self.assertAlmostEqual(self.node.current_y, 2.0)

    def test_odom_timestamp_rollback_reanchors_without_pose_jump(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 10.0))
        self.node.odom_callback(make_odom(0.1, 0.0, 0.0, 11.0))
        before = (self.node.current_x, self.node.current_y)

        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(self.node.current_x, before[0])
        self.assertAlmostEqual(self.node.current_y, before[1])

        self.node.odom_callback(make_odom(0.1, 0.0, 0.0, 2.0))
        initial_yaw = math.atan2(0.56, 6.27)
        self.assertAlmostEqual(
            self.node.current_x, before[0] + 0.1 * math.cos(initial_yaw))
        self.assertAlmostEqual(
            self.node.current_y, before[1] + 0.1 * math.sin(initial_yaw))

    def test_large_odom_jump_reanchors_without_teleporting_map_pose(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 1.0))
        self.node.odom_callback(make_odom(0.1, 0.0, 0.0, 2.0))
        before = (self.node.current_x, self.node.current_y)

        self.node.odom_callback(make_odom(10.0, 0.0, 0.0, 3.0))

        self.assertAlmostEqual(self.node.current_x, before[0])
        self.assertAlmostEqual(self.node.current_y, before[1])
        self.node.odom_callback(make_odom(10.1, 0.0, 0.0, 4.0))
        initial_yaw = math.atan2(0.56, 6.27)
        self.assertAlmostEqual(
            self.node.current_x, before[0] + 0.1 * math.cos(initial_yaw))
        self.assertAlmostEqual(
            self.node.current_y, before[1] + 0.1 * math.sin(initial_yaw))

    def test_large_odom_angle_jump_reanchors_without_rotating_map_pose(self):
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0, 1.0))
        self.node.odom_callback(make_odom(0.0, 0.0, 0.1, 2.0))
        before_yaw = self.node.current_yaw

        self.node.odom_callback(make_odom(0.0, 0.0, 2.0, 3.0))
        self.assertAlmostEqual(self.node.current_yaw, before_yaw)

        self.node.odom_callback(make_odom(0.0, 0.0, 2.1, 4.0))
        self.assertAlmostEqual(
            self.node.current_yaw,
            (before_yaw + 0.1 + math.pi) % (2.0 * math.pi) - math.pi,
        )

    def test_future_pose_timestamp_stops_after_clock_rollback(self):
        result = self.node.set_parameters([
            Parameter('pose_timeout', Parameter.Type.DOUBLE, 1.0)
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.scan_callback(make_scan())
        self.set_drive_enabled(True)
        self.assertGreater(self.tick()[0], 0.0)

        self.node.last_pose_time = (
            self.node.get_clock().now() + Duration(seconds=2.0))

        self.assertEqual(self.tick(), (0.0, 0.0))

    def test_final_goal_uses_immediate_stop(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())
        self.node.current_x = 6.27
        self.node.current_y = 0.56
        self.node.current_linear = 0.10

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertTrue(self.node.completed)

    def test_bypass_without_visible_wall_steers_toward_goal(self):
        self.node.scan_callback(make_scan())
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_started = now - Duration(seconds=2.0)
        self.node.avoidance_hit_distance = 5.0
        self.node.turn_direction = 1.0

        linear, angular = self.node.adjust_command_for_environment(
            desired_linear=0.10,
            desired_angular=0.0,
            distance=5.0,
            alpha=0.20,
            goal_x=6.27,
            goal_y=0.56,
            now=now,
        )

        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertLess(angular, 0.20)

    def test_turn_bypass_track_state_sequence(self):
        self.node.scan_callback(make_scan())
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'TURN'
        self.node.turn_direction = 1.0
        self.node.turn_start_yaw = 0.0
        self.node.current_yaw = 0.60
        self.node.avoidance_started = now - Duration(seconds=2.0)
        self.node.avoidance_hit_distance = 6.0
        self.node.clear_since = now - Duration(seconds=0.5)

        self.node.adjust_command_for_environment(
            0.10, 0.0, 5.80, 0.0, 6.27, 0.56, now)
        self.assertEqual(self.node.avoidance_state, 'BYPASS')

        linear, angular = self.node.adjust_command_for_environment(
            0.10, 0.0, 5.80, 0.0, 6.27, 0.56, now)
        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertAlmostEqual(linear, 0.10)
        self.assertAlmostEqual(angular, 0.0)

    def test_invalid_distance_order_is_rejected(self):
        result = self.node.set_parameters([
            Parameter(
                'front_stop_distance',
                Parameter.Type.DOUBLE,
                0.70,
            )
        ])[0]
        self.assertFalse(result.successful)


if __name__ == '__main__':
    unittest.main()
