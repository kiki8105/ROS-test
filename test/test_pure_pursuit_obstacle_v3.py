import math
import unittest

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from my_turtle_pkg.pure_pursuit_obstacle_v3 import (
    astar_path,
    gap_clearance_profile,
    grid_to_world,
    GridSpec,
    inflate_cell_set,
    minimum_radial_clearance,
    nearest_free_cell,
    nearest_reachable_free_path,
    occupancy_grid_to_blocked,
    parse_flat_waypoints,
    propagate_odom_pose,
    prune_ttl_cells,
    pure_pursuit_arc_is_free,
    PurePursuitObstacleV3,
    rear_body_clearance,
    refresh_ttl_cells,
    reverse_pure_pursuit_command,
    scan_hit_cells,
    select_collision_free_lookahead,
    select_lookahead_point,
    select_open_gap_heading,
    validate_mission_waypoints,
    WAFFLE_PI_FRONT_X,
    WAFFLE_PI_LIDAR_X,
    WAFFLE_PI_REAR_X,
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

    def test_open_gap_heading_prefers_wide_path_near_corridor(self):
        scan = make_scan(default=0.45)
        # A body-width opening centred near +45 degrees.  All remaining
        # headings leave only about 0.22m beyond the body radius.
        paint_sector(scan, 45.0, 32.0, 2.0)
        profile = gap_clearance_profile(
            scan,
            angle_step=math.radians(5.0),
            sector_half_width=math.radians(15.0),
            corridor_half_width=0.20,
            body_radius=0.23,
            minimum_hits=3,
        )

        selected = select_open_gap_heading(
            [profile, profile, profile],
            path_alpha=0.0,
            minimum_clearance=0.38,
            max_path_deviation=2.10,
            heading_weight=0.50,
            clearance_score_cap=1.50,
        )

        self.assertIsNotNone(selected)
        heading, clearance = selected
        self.assertGreater(heading, math.radians(25.0))
        self.assertLess(heading, math.radians(65.0))
        self.assertGreater(clearance, 1.0)

    def test_open_gap_heading_uses_multi_scan_median(self):
        angles = [(-0.5, 0.2), (0.0, 1.2), (0.5, 0.2)]
        noisy_once = [(-0.5, 0.2), (0.0, 0.1), (0.5, 0.2)]

        selected = select_open_gap_heading(
            [angles, noisy_once, angles],
            path_alpha=0.0,
            minimum_clearance=0.38,
            max_path_deviation=1.0,
            heading_weight=0.5,
            clearance_score_cap=1.5,
        )

        self.assertEqual(selected, (0.0, 1.2))

    def test_open_gap_heading_keeps_route_priority_over_wider_other_side(self):
        # This reproduces the field log: the A* line was to the right
        # (-0.38rad), but the former clearance-first score selected the much
        # wider left opening (+0.65rad).
        profile = [(-0.55, 0.42), (0.65, 1.39)]

        selected = select_open_gap_heading(
            [profile, profile, profile],
            path_alpha=-0.38,
            minimum_clearance=0.38,
            max_path_deviation=1.40,
            heading_weight=0.50,
            clearance_score_cap=1.50,
        )

        self.assertEqual(selected, (-0.55, 0.42))

    def test_open_gap_heading_rejects_candidates_behind_astar_direction(self):
        profile = [(-1.55, 3.5), (1.50, 3.5)]

        selected = select_open_gap_heading(
            [profile, profile, profile],
            path_alpha=0.0,
            minimum_clearance=0.38,
            max_path_deviation=1.40,
            heading_weight=0.50,
            clearance_score_cap=1.50,
        )

        self.assertIsNone(selected)

    def test_flat_waypoints_reject_odd_length_input(self):
        self.assertEqual(
            parse_flat_waypoints([1.0, 2.0, -3.0, 4.5]),
            [(1.0, 2.0), (-3.0, 4.5)],
        )
        with self.assertRaisesRegex(ValueError, '짝수'):
            parse_flat_waypoints([1.0, 2.0, 3.0])

    def test_scan_hit_cells_rotates_base_scan_hits_into_map(self):
        spec = GridSpec(30, 30, 1.0, -15.0, -15.0, 0.0)
        scan = LaserScan()
        scan.angle_min = 0.0
        scan.angle_increment = 0.1
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [2.0]

        # Robot yaw=+90deg turns a two-metre forward hit into map +Y.
        cells = scan_hit_cells(
            scan,
            robot_pose=(0.0, 0.0, math.pi / 2.0),
            spec=spec,
            min_range=0.1,
            max_range=5.0,
            lidar_x=0.0,
        )

        self.assertEqual(cells, {(15, 17)})

    def test_scan_hit_cells_filters_invalid_and_out_of_band_ranges(self):
        spec = GridSpec(30, 30, 1.0, -15.0, -15.0, 0.0)

        def projected(raw_range):
            scan = LaserScan()
            scan.angle_min = 0.0
            scan.angle_increment = 0.1
            scan.range_min = 0.1
            scan.range_max = 10.0
            scan.ranges = [raw_range]
            return scan_hit_cells(
                scan,
                robot_pose=(0.0, 0.0, 0.0),
                spec=spec,
                min_range=0.6,
                max_range=3.0,
                lidar_x=0.0,
            )

        for rejected in (math.inf, math.nan, 0.5, 3.1, 10.0):
            self.assertEqual(projected(rejected), set())
        self.assertEqual(projected(2.0), {(17, 15)})

    def test_ttl_refresh_and_prune_use_expiry_as_closed_boundary(self):
        expiry_by_cell = {}
        cell = (3, 4)

        self.assertTrue(refresh_ttl_cells(
            expiry_by_cell, {cell}, now_ns=100, ttl_ns=20))
        self.assertEqual(expiry_by_cell[cell], 120)

        # Refreshing an existing hit extends its life without growing the set.
        self.assertFalse(refresh_ttl_cells(
            expiry_by_cell, {cell}, now_ns=110, ttl_ns=20))
        self.assertEqual(expiry_by_cell[cell], 130)
        self.assertFalse(prune_ttl_cells(expiry_by_cell, now_ns=129))
        self.assertIn(cell, expiry_by_cell)

        # An obstacle expires exactly when now reaches its expiry timestamp.
        self.assertTrue(prune_ttl_cells(expiry_by_cell, now_ns=130))
        self.assertNotIn(cell, expiry_by_cell)

    def test_inflate_cell_set_uses_metric_radius_and_map_bounds(self):
        spec = GridSpec(3, 3, 1.0, 0.0, 0.0, 0.0)
        self.assertEqual(
            inflate_cell_set({(0, 0)}, spec, radius=1.0),
            {(0, 0), (1, 0), (0, 1)},
        )

    def test_inflated_start_escape_never_snaps_through_raw_wall(self):
        spec = GridSpec(7, 7, 0.1, 0.0, 0.0, 0.0)
        raw = [False] * 49
        # A full vertical wall separates the start from geometrically close
        # free cells on the opposite side.
        for grid_y in range(7):
            raw[grid_y * 7 + 3] = True
        static = [True] * 49
        static[3 * 7 + 5] = False

        self.assertIsNone(nearest_reachable_free_path(
            static, raw, spec, (2, 3), 0.5))

    def test_close_dynamic_hit_remains_in_projection_band(self):
        spec = GridSpec(40, 40, 0.1, -2.0, -2.0, 0.0)
        scan = LaserScan()
        scan.angle_min = 0.0
        scan.angle_increment = 0.1
        scan.range_min = 0.12
        scan.range_max = 10.0
        scan.ranges = [0.25]

        self.assertTrue(scan_hit_cells(
            scan,
            robot_pose=(0.0, 0.0, 0.0),
            spec=spec,
            min_range=0.20,
            max_range=2.0,
        ))

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

    def test_radial_clearance_requires_configured_number_of_hits(self):
        scan = make_scan()
        paint_sector(scan, 90.0, 0.1, 0.16)

        self.assertLess(minimum_radial_clearance(scan), 0.23)
        self.assertTrue(math.isinf(
            minimum_radial_clearance(scan, min_hits=3)))

    def test_rear_clearance_is_measured_from_body_rear_edge(self):
        scan = make_scan()
        expected_gap = 0.12
        raw_range = expected_gap + WAFFLE_PI_LIDAR_X - WAFFLE_PI_REAR_X
        paint_sector(scan, 180.0, 4.0, raw_range)

        self.assertAlmostEqual(
            rear_body_clearance(scan, min_hits=3),
            expected_gap,
            delta=0.002,
        )

    def test_reverse_pure_pursuit_uses_reverse_heading_and_correct_sign(self):
        linear, angular, _ = reverse_pure_pursuit_command(
            0.0, 0.0, 0.0, -1.0, 0.10, 0.03, 0.25)
        self.assertLess(linear, 0.0)
        self.assertLess(angular, 0.0)

        linear, angular, _ = reverse_pure_pursuit_command(
            0.0, 0.0, 0.0, -1.0, -0.10, 0.03, 0.25)
        self.assertLess(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_predicted_pure_pursuit_arc_detects_blocked_cell(self):
        spec = GridSpec(12, 12, 1.0, -5.0, -5.0, 0.0)
        blocked = [False] * (spec.width * spec.height)
        self.assertTrue(pure_pursuit_arc_is_free(
            blocked, spec, 0.0, 0.0, 0.0, 2.0, math.pi / 4.0))

        # The midpoint of this quarter-circle is near world (1.0, 0.41).
        blocked[5 * spec.width + 6] = True
        self.assertFalse(pure_pursuit_arc_is_free(
            blocked, spec, 0.0, 0.0, 0.0, 2.0, math.pi / 4.0))

    def test_straight_arc_does_not_add_false_diagonal_side_cell(self):
        spec = GridSpec(240, 240, 1.0, 0.0, 0.0, 0.0)
        blocked = [False] * (spec.width * spec.height)
        blocked[189 * spec.width + 190] = True
        start = grid_to_world(167, 139, spec)
        target = grid_to_world(191, 193, spec)
        delta_x = target[0] - start[0]
        delta_y = target[1] - start[1]

        self.assertTrue(pure_pursuit_arc_is_free(
            blocked,
            spec,
            start[0],
            start[1],
            math.atan2(delta_y, delta_x),
            math.hypot(delta_x, delta_y),
            0.0,
        ))

    def test_blocked_corner_arc_selects_shorter_safe_lookahead(self):
        spec = GridSpec(120, 80, 0.02, -0.5, -0.5, 0.0)
        blocked = [False] * (spec.width * spec.height)
        blocked[26 * spec.width + 74] = True
        path = [
            (0.0, 0.0),
            (1.0, 0.0),
            (
                1.0 + 0.5 * math.cos(math.radians(20.0)),
                0.5 * math.sin(math.radians(20.0)),
            ),
        ]
        x, y, yaw = 0.75, 0.0, 0.0
        target = select_lookahead_point(
            x, y, path[0], path, 1, 0.40)
        distance = math.hypot(target[0] - x, target[1] - y)
        alpha = math.atan2(target[1] - y, target[0] - x) - yaw
        self.assertFalse(pure_pursuit_arc_is_free(
            blocked, spec, x, y, yaw, distance, alpha))

        recovery = select_collision_free_lookahead(
            blocked,
            spec,
            x,
            y,
            yaw,
            path[0],
            path,
            1,
            preferred_distance=0.40,
            minimum_distance=0.12,
        )

        self.assertIsNotNone(recovery)
        recovery_distance, recovery_alpha, used_lookahead = recovery
        self.assertAlmostEqual(used_lookahead, 0.38)
        self.assertTrue(pure_pursuit_arc_is_free(
            blocked,
            spec,
            x,
            y,
            yaw,
            recovery_distance,
            recovery_alpha,
        ))

    def test_close_saved_corner_is_not_selected_as_recovery_target(self):
        spec = GridSpec(120, 80, 0.02, -0.5, -0.5, 0.0)
        blocked = [False] * (spec.width * spec.height)
        blocked[25 * spec.width + 76] = True
        path = [
            (0.0, 0.0),
            (1.0, 0.0),
            (
                1.0 + 0.5 * math.cos(math.radians(20.0)),
                0.5 * math.sin(math.radians(20.0)),
            ),
        ]

        recovery = select_collision_free_lookahead(
            blocked,
            spec,
            x=0.90,
            y=0.0,
            yaw=0.0,
            segment_start=path[1],
            path=path,
            waypoint_index=2,
            preferred_distance=0.40,
            minimum_distance=0.12,
        )

        self.assertIsNone(recovery)

    def test_recovery_accepts_one_map_cell_lookahead(self):
        spec = GridSpec(20, 5, 0.05, 0.0, -0.1, 0.0)
        blocked = [False] * (spec.width * spec.height)
        obstacle = world_to_grid(0.10, 0.0, spec)
        blocked[obstacle[1] * spec.width + obstacle[0]] = True
        path = [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.10, 0.0),
            (0.20, 0.0),
            (0.40, 0.0),
        ]

        self.assertIsNone(select_collision_free_lookahead(
            blocked, spec, 0.0, 0.0, 0.0, path[0], path, 1,
            preferred_distance=0.40,
            minimum_distance=0.12,
        ))
        recovery = select_collision_free_lookahead(
            blocked, spec, 0.0, 0.0, 0.0, path[0], path, 1,
            preferred_distance=0.40,
            minimum_distance=0.05,
        )

        self.assertIsNotNone(recovery)
        distance, alpha, used_lookahead = recovery
        self.assertAlmostEqual(distance, 0.05)
        self.assertAlmostEqual(alpha, 0.0)
        self.assertAlmostEqual(used_lookahead, 0.05)

    def test_right_angle_is_preserved_but_astar_staircase_is_smoothed(self):
        right_angle_path = [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.05, 0.40),
        ]
        target = select_lookahead_point(
            0.0, 0.0, right_angle_path[0], right_angle_path, 1, 0.40,
            corner_angle_limit=1.0,
        )
        self.assertEqual(target, right_angle_path[1])

        staircase_path = [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.10, 0.05),
            (0.15, 0.05),
            (0.20, 0.10),
            (0.40, 0.10),
        ]
        target = select_lookahead_point(
            0.0, 0.0, staircase_path[0], staircase_path, 1, 0.30,
            corner_angle_limit=1.0,
        )
        self.assertGreater(math.hypot(*target), 0.20)


class IntegratedNodeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = PurePursuitObstacleV3()
        self.node.timer.cancel()
        self.gap_recovery_default = bool(
            self.node.get_parameter('gap_recovery_enabled').value)
        # Legacy tests below intentionally exercise TURN/BYPASS/OFFSET.  The
        # production default is tested separately and focused GAP tests enable
        # the new state machine explicitly.
        result = self.node.set_parameters([
            Parameter(
                'gap_recovery_enabled', Parameter.Type.BOOL, False),
        ])[0]
        self.assertTrue(result.successful, result.reason)
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

    def enable_gap_recovery(self):
        result = self.node.set_parameters([
            Parameter(
                'gap_recovery_enabled', Parameter.Type.BOOL, True),
        ])[0]
        self.assertTrue(result.successful, result.reason)

    def test_gap_recovery_is_enabled_by_default(self):
        self.assertTrue(self.gap_recovery_default)

    def test_front_obstacle_enters_stopped_gap_scan_when_enabled(self):
        self.prepare_plan()
        self.enable_gap_recovery()
        self.set_enabled(True)
        self.node.scan_callback(scan_with_front_gap(0.25, left=2.0, right=0.5))

        command = self.tick()

        self.assertEqual(self.node.avoidance_state, 'GAP_SCAN')
        self.assertEqual(command, (0.0, 0.0))
        self.assertNotIn(self.node.avoidance_state, ('TURN', 'BYPASS'))

    def test_gap_scan_requires_fresh_scans_and_resets_old_turn_command(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())
        self.node.odom_callback(make_odom())
        self.node.goal_callback(make_goal())
        self.tick()
        self.enable_gap_recovery()

        opening = make_scan(default=0.45)
        paint_sector(opening, 45.0, 32.0, 2.0)
        self.node.scan_callback(opening)
        metrics = self.node.scan_metrics(0.0)
        self.node.current_angular = -0.24
        self.assertTrue(self.node.start_gap_recovery(
            metrics, self.node.get_clock().now(), 0.0, 'test'))
        self.assertEqual(self.node.avoidance_state, 'GAP_SCAN')
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))
        self.assertEqual(self.node.current_angular, 0.0)

        # Reusing the triggering scan does not count as a stopped sample.
        self.node.gap_scan_command(metrics, self.node.get_clock().now())
        self.assertEqual(len(self.node.gap_profiles), 0)
        for expected in (1, 2, 3):
            self.node.scan_callback(opening)
            metrics = self.node.scan_metrics(0.0)
            self.node.gap_scan_command(metrics, self.node.get_clock().now())
            if expected < 3:
                self.assertEqual(self.node.avoidance_state, 'GAP_SCAN')
                self.assertEqual(len(self.node.gap_profiles), expected)
        self.assertEqual(self.node.avoidance_state, 'GAP_ALIGN')
        self.assertIsNotNone(self.node.gap_target_yaw)
        self.assertGreater(self.node.gap_target_relative, 0.0)

    def test_gap_align_is_latched_slow_and_replans_after_entry(self):
        result = self.node.set_parameters([
            Parameter(
                'dynamic_obstacles_enabled', Parameter.Type.BOOL, False),
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())
        self.node.odom_callback(make_odom())
        self.node.goal_callback(make_goal())
        self.tick()
        self.enable_gap_recovery()

        self.node.gap_started = self.node.get_clock().now()
        self.node.gap_target_yaw = 0.60
        self.node.gap_target_relative = 0.60
        self.node.gap_turn_sign = 1.0
        self.node.avoidance_state = 'GAP_ALIGN'
        self.node.avoidance_active = True
        open_scan = make_scan()
        self.node.scan_callback(open_scan)
        metrics = self.node.scan_metrics(0.0)

        self.node.current_yaw = 0.0
        self.node.gap_align_command(
            metrics, self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.avoidance_state, 'GAP_ALIGN')
        self.assertEqual(self.twists.commands[-1][0], 0.0)
        self.assertGreater(self.twists.commands[-1][1], 0.0)
        self.assertLessEqual(abs(self.twists.commands[-1][1]), 0.15)
        latched_target = self.node.gap_target_yaw

        # New scans cannot switch the already selected side while aligning.
        opposite = make_scan(default=0.45)
        paint_sector(opposite, -60.0, 35.0, 3.0)
        self.node.scan_callback(opposite)
        self.node.current_yaw = 0.30
        self.node.gap_align_command(
            self.node.scan_metrics(0.0), self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.gap_target_yaw, latched_target)
        self.assertGreaterEqual(self.twists.commands[-1][1], 0.0)

        # Once aligned, stop first and enter the opening on the next tick.
        self.node.scan_callback(open_scan)
        self.node.current_yaw = latched_target
        self.node.latest_odom_pose = (0.0, 0.0, latched_target)
        self.node.gap_align_command(
            self.node.scan_metrics(0.0), self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.avoidance_state, 'GAP_ADVANCE')
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

        old_plan_count = self.node.plan_count
        self.node.latest_odom_pose = (0.20, 0.0, latched_target)
        self.node.gap_advance_command(
            self.node.scan_metrics(0.0), self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.plan_count, old_plan_count + 1)
        self.assertNotIn(self.node.avoidance_state, ('TURN', 'BYPASS'))

    def prepare_plan(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose())
        self.node.goal_callback(make_goal())
        self.tick()
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertGreaterEqual(len(self.node.path), 2)

    def prepare_backtrack_case(self):
        """Prepare a forward odom trail while the robot faces away from goal."""
        # Most focused BACKTRACK tests exercise the state transition with a
        # short confirmation streak; production defaults use six scans.
        result = self.node.set_parameters([
            Parameter('spin_block_confirm_scans', Parameter.Type.INTEGER, 3),
        ])[0]
        self.assertTrue(result.successful, result.reason)
        self.node.map_callback(make_map())
        self.node.odom_callback(make_odom(0.0, 0.0, math.pi))
        self.node.pose_callback(make_pose(0.0, 0.0, math.pi))
        self.node.goal_callback(make_goal())
        self.tick()
        self.assertEqual(self.node.planner_state, 'READY')

        self.node.current_linear = 0.08
        for x in (-0.05, -0.10, -0.15, -0.20, -0.25):
            self.node.odom_callback(make_odom(x, 0.0, math.pi))
        self.node.current_linear = 0.0
        self.set_enabled(True)

        scan = make_scan()
        paint_sector(scan, 90.0, 5.0, 0.16)
        return scan

    def prepare_corner_tracking(self, blocked):
        spec = GridSpec(120, 80, 0.02, -0.5, -0.5, 0.0)
        path = [
            (0.0, 0.0),
            (1.0, 0.0),
            (
                1.0 + 0.5 * math.cos(math.radians(20.0)),
                0.5 * math.sin(math.radians(20.0)),
            ),
        ]
        self.node.blocked = blocked
        self.node.raw_blocked = list(blocked)
        self.node.grid_spec = spec
        self.node.costmap_dirty = False
        self.node.current_x = 0.75
        self.node.current_y = 0.0
        self.node.current_yaw = 0.0
        self.node.is_localized = True
        self.node.last_pose_time = self.node.get_clock().now()
        self.node.requested_goal = path[-1]
        self.node.active_goal = path[-1]
        self.node.goal_source = 'EXTERNAL'
        self.node.path = path
        self.node.current_waypoint_index = 1
        self.node.segment_start = path[0]
        self.node.replan_requested = False
        self.node.last_plan_time = self.node.get_clock().now()
        self.node.planner_state = 'READY'
        self.node.completed = False
        self.node.mission_active = False
        self.node.scan_callback(make_scan())
        self.set_enabled(True)

    def test_disabled_node_plans_and_publishes_path_but_stays_stopped(self):
        self.prepare_plan()

        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))
        self.assertGreater(len(self.paths.messages[-1].poses), 1)
        self.assertEqual(self.paths.messages[-1].header.frame_id, 'map')
        self.assertAlmostEqual(self.node.path[0][0], 0.0)
        self.assertAlmostEqual(self.node.path[-1][0], 1.5)

    def test_planner_keeps_dense_astar_cells_for_safe_tracking(self):
        self.prepare_plan()

        self.assertEqual(len(self.node.path), len(self.node.path_cells))
        self.assertGreater(len(self.node.path_cells), 2)
        for current_cell, next_cell in zip(
                self.node.path_cells, self.node.path_cells[1:]):
            self.assertLessEqual(abs(next_cell[0] - current_cell[0]), 1)
            self.assertLessEqual(abs(next_cell[1] - current_cell[1]), 1)

    def test_dense_astar_staircase_keeps_metric_lookahead(self):
        self.node.current_x = 0.0
        self.node.current_y = 0.0
        self.node.current_yaw = 0.0
        self.node.path = [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.2, 0.1),
            (0.3, 0.1),
            (0.4, 0.2),
            (0.5, 0.2),
        ]
        self.node.segment_start = self.node.path[0]
        self.node.current_waypoint_index = 1
        self.node.active_goal = self.node.path[-1]

        _, target_distance, _ = self.node.path_geometry()

        self.assertGreater(target_distance, 0.30)

    def test_angular_limit_preserves_checked_pure_pursuit_curvature(self):
        target_distance = 0.05
        alpha = 0.29
        expected_curvature = (
            2.0 * math.sin(alpha) / target_distance)

        linear, angular, rotate_only = self.node.path_command(
            goal_distance=5.0,
            target_distance=target_distance,
            alpha=alpha,
        )

        self.assertFalse(rotate_only)
        self.assertLess(linear, 0.08)
        self.assertLessEqual(abs(angular), 0.60)
        self.assertAlmostEqual(angular / linear, expected_curvature)

    def test_return_home_tolerance_is_also_used_for_braking(self):
        result = self.node.set_parameters([
            Parameter(
                'return_home_tolerance', Parameter.Type.DOUBLE, 0.05),
        ])[0]
        self.assertTrue(result.successful, result.reason)

        # 0.15 m is inside the ordinary 0.20 m goal tolerance, but remains
        # outside the stricter 0.05 m home tolerance.  A RETURN_HOME leg must
        # therefore keep approaching rather than braking to zero early.
        self.node.goal_source = 'RETURN_HOME'
        home_linear, _, _ = self.node.path_command(
            goal_distance=0.15,
            target_distance=0.40,
            alpha=0.0,
        )
        self.assertGreater(home_linear, 0.0)

        self.node.goal_source = 'EXTERNAL'
        ordinary_linear, _, _ = self.node.path_command(
            goal_distance=0.15,
            target_distance=0.40,
            alpha=0.0,
        )
        self.assertEqual(ordinary_linear, 0.0)

    def test_new_external_goal_clears_old_dynamic_replan_pending(self):
        self.node.dynamic_replan_pending = True
        self.node.requested_goal = (0.5, 0.0)

        self.node.goal_callback(make_goal(1.2, -0.4))

        self.assertFalse(self.node.dynamic_replan_pending)
        self.assertEqual(self.node.requested_goal, (1.2, -0.4))
        self.assertTrue(self.node.replan_requested)

    def test_lidar_hit_near_mapped_wall_is_not_a_dynamic_obstacle(self):
        message = make_map()
        # Robot cell is (10, 10).  A measured endpoint lands at (15, 10),
        # while the mapped wall is five 0.1 m cells farther at (20, 10).
        message.data[10 * message.info.width + 20] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose())
        scan = LaserScan()
        scan.header.frame_id = 'base_scan'
        scan.angle_min = 0.0
        scan.angle_increment = 0.1
        scan.range_min = 0.12
        scan.range_max = 10.0
        scan.ranges = [0.65]

        self.node.scan_callback(scan)

        self.assertFalse(self.node.dynamic_expiry)

    def test_lidar_hit_far_from_mapped_wall_stays_dynamic(self):
        message = make_map()
        message.data[10 * message.info.width + 30] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose())
        scan = LaserScan()
        scan.header.frame_id = 'base_scan'
        scan.angle_min = 0.0
        scan.angle_increment = 0.1
        scan.range_min = 0.12
        scan.range_max = 10.0
        scan.ranges = [0.65]

        self.node.scan_callback(scan)

        self.assertTrue(self.node.dynamic_expiry)

    def test_avoidance_timeout_uses_verified_backtrack_before_blocking(self):
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'TURN'
        timeout = float(
            self.node.get_parameter('avoidance_max_duration').value)
        self.node.avoidance_started = now - Duration(seconds=timeout + 1.0)
        self.node.avoidance_direction_switches = int(
            self.node.get_parameter(
                'avoidance_direction_retry_limit').value)
        calls = []
        self.node.start_unwind_turn = (
            lambda stamp: calls.append('unwind') or False)
        self.node.start_backtrack = (
            lambda metrics, stamp: calls.append('backtrack') or True)

        self.node.turn_command({'radial': 1.0}, now, 0.1)

        self.assertEqual(calls, ['unwind', 'backtrack'])

    def test_clear_scan_and_enable_publish_forward_cmd_vel(self):
        self.prepare_plan()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)

        linear, angular = self.tick()

        self.assertGreater(linear, 0.0)
        # 실제 시작 자세와 첫 A* 셀 중심 사이의 최대 반 셀 오프셋은
        # 작은 조향 보정을 만들 수 있다.
        self.assertAlmostEqual(angular, 0.0, delta=0.10)
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

    def test_front_obstacle_turn_enters_backtrack_after_six_scans(self):
        """Exercise the real TRACK -> TURN -> BACKTRACK transition."""
        self.node.map_callback(make_map())
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.pose_callback(make_pose(0.0, 0.0, 0.0))
        self.node.goal_callback(make_goal())
        self.tick()

        self.node.current_linear = 0.08
        for x in (0.05, 0.10, 0.15, 0.20, 0.25):
            self.node.odom_callback(make_odom(x, 0.0, 0.0))
        self.node.current_linear = 0.0
        self.set_enabled(True)

        close_front = scan_with_front_gap(0.05, left=2.0, right=0.55)
        for expected_count in range(1, 6):
            self.node.scan_callback(close_front)
            self.assertEqual(self.tick(), (0.0, 0.0))
            self.assertEqual(self.node.avoidance_state, 'TURN')
            self.assertEqual(self.node.spin_block_count, expected_count)

        self.node.scan_callback(close_front)
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertGreaterEqual(self.node.backtrack_planned_distance, 0.20)

    def test_spin_confirmation_restarts_after_skipped_scan_sequence(self):
        metrics = {'radial': 0.10}
        now = self.node.get_clock().now()

        self.node.scan_sequence = 10
        self.assertTrue(self.node.handle_spin_block(metrics, now))
        self.assertEqual(self.node.spin_block_count, 1)

        # Sequence 11 represents an intervening scan handled by a different
        # control branch.  Sequence 12 must start a fresh streak.
        self.node.scan_sequence = 12
        self.assertTrue(self.node.handle_spin_block(metrics, now))
        self.assertEqual(self.node.spin_block_count, 1)
        self.assertEqual(self.node.spin_block_last_scan_sequence, 12)

    def test_forward_motion_clears_spin_confirmation(self):
        self.node.spin_block_count = 1
        self.node.spin_block_last_scan_sequence = 7

        self.node.publish_motion(0.05, 0.0, 0.1)

        self.assertEqual(self.node.spin_block_count, 0)
        self.assertEqual(self.node.spin_block_last_scan_sequence, -1)

    def test_backtrack_path_stops_before_curved_older_trail(self):
        self.node.latest_odom_pose = (0.0, 0.0, 0.0)
        self.node.forward_trail.clear()
        self.node.forward_trail.extend([
            (-0.20, 0.08, 0.40),
            (-0.15, 0.05, 0.30),
            (-0.10, 0.01, 0.05),
            (-0.05, 0.00, 0.00),
        ])

        path, distance = self.node._build_backtrack_path()

        self.assertGreaterEqual(distance, 0.10)
        self.assertLess(distance, 0.15)
        self.assertEqual(len(path), 3)
        self.assertAlmostEqual(path[-1][0], -0.10)
        self.assertAlmostEqual(path[-1][1], 0.0)

    def test_backtrack_uses_curved_breadcrumbs_if_straight_tail_is_short(self):
        self.node.latest_odom_pose = (0.0, 0.0, 0.0)
        self.node.forward_trail.clear()
        self.node.forward_trail.extend([
            (-0.08, 0.04, 0.40),
            (-0.04, 0.00, 0.00),
        ])

        path, distance = self.node._build_backtrack_path()

        self.assertGreaterEqual(distance, 0.06)
        self.assertGreaterEqual(len(path), 3)
        self.assertAlmostEqual(path[1][0], -0.04)
        self.assertAlmostEqual(path[-1][1], 0.04)

    def test_second_short_backtrack_is_allowed_but_third_is_blocked(self):
        """A successful replan may need one more bounded local recovery."""
        result = self.node.set_parameters([
            Parameter('backtrack_retry_limit', Parameter.Type.INTEGER, 2),
        ])[0]
        self.assertTrue(result.successful, result.reason)
        now = self.node.get_clock().now()
        self.node.backtrack_attempts = 1
        self.node.backtrack_trigger_pose = (0.0, 0.0)
        self.node.current_x = 0.05
        self.node.current_y = 0.0
        self.node.latest_odom_pose = (0.0, 0.0, 0.0)
        self.node.last_odom_time = now
        self.node.forward_trail.clear()
        self.node.forward_trail.extend([
            (-0.08, 0.0, 0.0),
            (-0.06, 0.0, 0.0),
            (-0.04, 0.0, 0.0),
            (-0.02, 0.0, 0.0),
        ])
        metrics = {
            'rear_range_count': 20,
            'rear': math.inf,
        }

        self.assertTrue(self.node.start_backtrack(metrics, now))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.backtrack_attempts, 2)
        self.assertGreaterEqual(self.node.backtrack_planned_distance, 0.06)

        self.node._clear_backtrack_path()
        self.node.avoidance_state = 'TRACK'
        self.assertFalse(self.node.start_backtrack(metrics, now))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_verified_odom_backtrack_is_not_vetoed_by_slam_map_error(self):
        self.prepare_backtrack_case()
        now = self.node.get_clock().now()
        metrics = {
            'rear_range_count': 20,
            'rear': math.inf,
            'radial': 1.0,
        }
        self.assertTrue(self.node.start_backtrack(metrics, now))

        # Model a badly shifted saved map that marks even the robot's recently
        # traversed breadcrumb as occupied.  Live rear/radial LiDAR and the
        # bounded odom trace must control this recovery, not stale map pixels.
        self.node.raw_blocked = [True] * len(self.node.raw_blocked)
        self.node.blocked = [True] * len(self.node.blocked)
        self.node.backtrack_command(metrics, now, 0.1)

        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertLess(self.twists.commands[-1][0], 0.0)

    def test_single_close_spin_scan_stops_without_permanent_block(self):
        close_scan = self.prepare_backtrack_case()
        self.node.scan_callback(close_scan)

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.spin_block_count, 1)

        self.node.scan_callback(make_scan())
        _, angular = self.tick()
        self.assertNotEqual(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.spin_block_count, 0)

    def test_confirmed_spin_block_backtracks_along_frozen_odom_trail(self):
        close_scan = self.prepare_backtrack_case()
        original_plan_count = self.node.plan_count
        original_waypoint = self.node.current_waypoint_index

        for _ in range(3):
            self.node.scan_callback(close_scan)
            command = self.tick()

        self.assertEqual(command, (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertGreaterEqual(self.node.backtrack_planned_distance, 0.20)

        self.node.scan_callback(close_scan)
        linear, angular = self.tick()
        self.assertLess(linear, 0.0)
        self.assertEqual(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.plan_count, original_plan_count)
        self.assertEqual(self.node.current_waypoint_index, original_waypoint)

        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        for expected_count in (1, 2):
            self.node.scan_callback(make_scan())
            self.assertEqual(self.tick(), (0.0, 0.0))
            self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
            self.assertEqual(
                self.node.backtrack_release_count, expected_count)
        self.node.scan_callback(make_scan())
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertEqual(self.node.plan_count, original_plan_count)
        self.assertEqual(len(self.node.forward_trail), 1)
        self.assertAlmostEqual(self.node.forward_trail[0][0], final[0])
        self.assertEqual(self.node.backtrack_path, [])

    def test_backtrack_release_resumes_open_original_turn_before_replan(self):
        """An open selected side must break the backtrack/forward loop."""
        close_scan = self.prepare_backtrack_case()
        original_plan_count = self.node.plan_count

        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.turn_direction, -1.0)

        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        for _ in range(3):
            self.node.scan_callback(make_scan())
            self.tick()

        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, -1.0)
        self.assertEqual(self.node.plan_count, original_plan_count)
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_backtrack_release_resumes_left_when_lidar_left_is_open(self):
        """The latest logged LEFT choice must survive reverse recovery."""
        self.prepare_backtrack_case()
        self.node.turn_direction = 1.0
        self.node.avoidance_state = 'BACKTRACK'
        self.node.backtrack_start_odom = tuple(self.node.latest_odom_pose)
        original_plan_count = self.node.plan_count
        metrics = {
            'radial': 0.29,
            'front': 0.30,
            'front_left': 1.20,
            'left': 1.00,
            'front_right': 0.30,
            'right': 0.25,
        }

        self.node._finish_backtrack(metrics, self.node.get_clock().now())

        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)
        self.assertEqual(self.node.plan_count, original_plan_count)
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_backtrack_release_requires_margin_and_three_fresh_scans(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')

        # Move 0.10 m along the frozen reverse trace.  One open scan used to
        # release BACKTRACK here; it must now start only a confirmation streak.
        partial = self.node.backtrack_path[2]
        self.node.odom_callback(make_odom(partial[0], partial[1], math.pi))
        self.node.scan_callback(make_scan())
        self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.backtrack_release_count, 1)

        # A new close scan cancels the incomplete release confirmation.
        self.node.scan_callback(close_scan)
        self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.backtrack_release_count, 0)

        for expected_count in (1, 2):
            self.node.scan_callback(make_scan())
            self.tick()
            self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
            self.assertEqual(
                self.node.backtrack_release_count, expected_count)
        self.node.scan_callback(make_scan())
        self.tick()
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.planner_state, 'READY')

    def test_early_release_discards_old_trace_and_resumes_turn(self):
        close_scan = self.prepare_backtrack_case()
        original_plan_count = self.node.plan_count
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')

        partial = self.node.backtrack_path[2]
        self.node.odom_callback(make_odom(partial[0], partial[1], math.pi))
        for _ in range(3):
            self.node.scan_callback(make_scan())
            self.tick()

        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertEqual(self.node.plan_count, original_plan_count)
        self.assertEqual(self.node.backtrack_attempts, 1)
        self.assertEqual(len(self.node.forward_trail), 1)
        self.assertEqual(self.node.backtrack_path, [])
        self.assertEqual(self.node.backtrack_planned_distance, 0.0)

    def test_backtrack_replan_waits_if_dynamic_layer_covers_new_start(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')

        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        start_cell = world_to_grid(
            self.node.current_x, self.node.current_y, self.node.grid_spec)
        start_index = (
            start_cell[1] * self.node.grid_spec.width + start_cell[0])
        self.assertFalse(self.node.static_blocked[start_index])
        self.node.blocked[start_index] = True

        self.node._finish_backtrack(
            {'radial': 1.0}, self.node.get_clock().now())

        self.assertEqual(self.node.avoidance_state, 'WAIT_DYNAMIC_CLEAR')
        self.assertEqual(self.node.planner_state, 'WAIT_DYNAMIC_CLEAR')
        self.assertFalse(self.node.mission_failed)
        self.assertEqual(self.node.path, [])

    def test_backtrack_release_rejects_dynamic_start_cell(self):
        self.node.map_callback(make_map(resolution=0.10))
        self.node.pose_callback(make_pose())
        start_cell = world_to_grid(0.0, 0.0, self.node.grid_spec)
        self.node.dynamic_inflated_cells = {start_cell}

        self.assertFalse(self.node.dynamic_start_cell_is_clear())
        self.assertFalse(self.node._backtrack_release_is_confirmed(
            {'radial': 1.0}, travelled=0.20))

    def test_wait_dynamic_clear_retries_on_unchanged_fresh_scan(self):
        self.prepare_plan()
        self.node.planner_state = 'WAIT_DYNAMIC_CLEAR'
        self.node.dynamic_replan_pending = False

        self.node.scan_callback(make_scan())

        self.assertTrue(self.node.dynamic_replan_pending)

    def test_forward_motion_discards_retained_backtrack_trace(self):
        self.node.backtrack_path = [(0.0, 0.0), (-0.10, 0.0)]
        self.node.backtrack_waypoint_index = 1
        self.node.backtrack_segment_start = (0.0, 0.0)
        self.node.avoidance_state = 'BYPASS'

        self.node.publish_motion(0.05, 0.0, 0.1)

        self.assertEqual(self.node.backtrack_path, [])
        self.assertEqual(self.node.backtrack_waypoint_index, 0)

    def test_path_end_needs_three_low_scans_before_blocking(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))

        self.node.scan_callback(close_scan)
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.spin_block_count, 1)

        # Re-evaluating the same scan must not count as another observation.
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.spin_block_count, 1)

        self.node.scan_callback(close_scan)
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.spin_block_count, 2)

        self.node.scan_callback(close_scan)
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_right_turn_backtrack_end_starts_left_offset_recovery(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.node.turn_direction = -1.0
        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        metrics = {
            'radial': 0.24,
            'rear': 1.0,
            'rear_range_count': 30,
            'front': 2.0,
            'front_left': 2.0,
            'left': 2.0,
            'front_right': 0.25,
            'right': 0.25,
        }

        # 0.24 m clears the 0.23 m minimum spin radius but not the 0.28 m
        # BACKTRACK release threshold.  Three fresh low-margin scans should
        # authorize a left S shift instead of terminal BLOCKED.
        for _ in range(3):
            self.node.scan_sequence += 1
            self.node.backtrack_command(
                metrics, self.node.get_clock().now(), 0.1)

        self.assertEqual(self.node.avoidance_state, 'OFFSET_TURN')
        self.assertEqual(self.node.offset_original_turn_direction, -1.0)
        self.assertEqual(self.node.offset_shift_direction, 1.0)
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_track_right_spin_latches_direction_through_backtrack_to_offset(self):
        """TRACK rotate-only must preserve its sign for later S recovery."""
        close_scan = self.prepare_backtrack_case()

        # Facing west while the A* goal is east produces a right-hand
        # rotate-only command.  No TURN state has populated turn_direction.
        self.assertEqual(self.node.turn_direction, 0.0)
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()

        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertEqual(self.node.turn_direction, -1.0)
        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        low_margin_left_open = {
            'radial': 0.24,
            'rear': 1.0,
            'rear_range_count': 30,
            'front': 2.0,
            'front_left': 2.0,
            'left': 2.0,
            'front_right': 0.25,
            'right': 0.25,
        }
        for _ in range(3):
            self.node.scan_sequence += 1
            self.node.backtrack_command(
                low_margin_left_open,
                self.node.get_clock().now(),
                0.1,
            )

        self.assertEqual(self.node.avoidance_state, 'OFFSET_TURN')
        self.assertEqual(self.node.offset_original_turn_direction, -1.0)
        self.assertEqual(self.node.offset_shift_direction, 1.0)

    def test_left_turn_blockage_selects_symmetric_right_offset(self):
        self.prepare_backtrack_case()
        self.node.turn_direction = 1.0
        metrics = {
            'radial': 0.24,
            'front': 2.0,
            'front_left': 0.25,
            'left': 0.25,
            'front_right': 2.0,
            'right': 2.0,
        }

        self.assertTrue(self.node.start_offset_recovery(
            metrics, self.node.get_clock().now()))
        self.assertEqual(self.node.avoidance_state, 'OFFSET_TURN')
        self.assertEqual(self.node.offset_original_turn_direction, 1.0)
        self.assertEqual(self.node.offset_shift_direction, -1.0)

    def test_offset_recovery_runs_turn_forward_realign_and_replans(self):
        self.prepare_backtrack_case()
        self.node.turn_direction = -1.0
        metrics = {
            'radial': 1.0,
            'front': 2.0,
            'front_left': 2.0,
            'left': 2.0,
            'front_right': 0.25,
            'right': 0.25,
        }
        original_plan_count = self.node.plan_count
        now = self.node.get_clock().now()
        self.assertTrue(self.node.start_offset_recovery(metrics, now))
        original_yaw = self.node.offset_original_yaw
        shifted_yaw = self.node.offset_target_yaw
        start_x, start_y, _ = self.node.latest_odom_pose

        self.node.odom_callback(make_odom(start_x, start_y, shifted_yaw))
        self.node.offset_recovery_command(
            metrics, self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.avoidance_state, 'OFFSET_FORWARD')

        distance = float(self.node.get_parameter(
            'offset_recovery_distance').value)
        end_x = start_x + distance * math.cos(shifted_yaw)
        end_y = start_y + distance * math.sin(shifted_yaw)
        self.node.odom_callback(make_odom(end_x, end_y, shifted_yaw))
        self.node.offset_recovery_command(
            metrics, self.node.get_clock().now(), 0.1)
        self.assertEqual(self.node.avoidance_state, 'OFFSET_REALIGN')

        self.node.odom_callback(make_odom(end_x, end_y, original_yaw))
        self.node.offset_recovery_command(
            metrics, self.node.get_clock().now(), 0.1)

        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertEqual(self.node.plan_count, original_plan_count + 1)
        self.assertGreaterEqual(len(self.node.path), 2)

    def test_offset_recovery_refuses_raw_map_collision(self):
        self.prepare_backtrack_case()
        self.node.turn_direction = -1.0
        current_x, current_y, current_yaw = self.node.latest_odom_pose
        target_yaw = current_yaw + float(self.node.get_parameter(
            'offset_recovery_turn_angle').value)
        distance = float(self.node.get_parameter(
            'offset_recovery_distance').value)
        blocked_world = (
            self.node.current_x + distance * math.cos(target_yaw),
            self.node.current_y + distance * math.sin(target_yaw),
        )
        blocked_cell = world_to_grid(
            blocked_world[0], blocked_world[1], self.node.grid_spec)
        self.assertIsNotNone(blocked_cell)
        self.node.raw_blocked[
            blocked_cell[1] * self.node.grid_spec.width
            + blocked_cell[0]
        ] = True
        metrics = {
            'radial': 0.24,
            'front': 2.0,
            'front_left': 2.0,
            'left': 2.0,
            'front_right': 0.25,
            'right': 0.25,
        }

        self.assertFalse(self.node.start_offset_recovery(
            metrics, self.node.get_clock().now()))
        self.assertNotEqual(self.node.avoidance_state, 'OFFSET_TURN')

    def test_short_valid_trace_can_release_after_reaching_its_end(self):
        close_scan = self.prepare_backtrack_case()
        # Keep only 0.08 m of straight forward history.  It satisfies the
        # 0.06 m entry requirement but is shorter than the 0.10 m guard used
        # only for early release.
        self.node.forward_trail.clear()
        self.node.forward_trail.extend([
            (-0.17, 0.0, math.pi),
            (-0.19, 0.0, math.pi),
            (-0.21, 0.0, math.pi),
            (-0.23, 0.0, math.pi),
            (-0.25, 0.0, math.pi),
        ])
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertAlmostEqual(
            self.node.backtrack_planned_distance, 0.08, delta=0.001)

        final = self.node.backtrack_path[-1]
        self.node.odom_callback(make_odom(final[0], final[1], math.pi))
        for expected_count in (1, 2):
            self.node.scan_callback(make_scan())
            self.assertEqual(self.tick(), (0.0, 0.0))
            self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
            self.assertEqual(
                self.node.backtrack_release_count, expected_count)
        self.node.scan_callback(make_scan())
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.planner_state, 'READY')

    def test_backtrack_stops_if_rear_corridor_becomes_blocked(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')

        rear_blocked = make_scan()
        rear_gap = 0.05
        raw_range = rear_gap + WAFFLE_PI_LIDAR_X - WAFFLE_PI_REAR_X
        paint_sector(rear_blocked, 180.0, 5.0, raw_range)
        self.node.scan_callback(rear_blocked)

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_backtrack_sensor_timeout_latches_blocked_instead_of_track(self):
        close_scan = self.prepare_backtrack_case()
        for _ in range(3):
            self.node.scan_callback(close_scan)
            self.tick()
        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')

        self.node.last_scan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')

    def test_heading_alignment_does_not_start_bypass_for_safe_front_gap(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.0, math.pi))
        self.node.goal_callback(make_goal())
        self.tick()
        self.node.scan_callback(
            scan_with_front_gap(0.36, left=2.0, right=0.55))
        self.set_enabled(True)

        linear, angular = self.tick()

        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)
        self.assertTrue(self.node.heading_align_active)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_close_front_aligns_to_an_open_path_ray_before_moving(self):
        self.prepare_plan()
        self.node.current_yaw = -0.50
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=2.0, right=2.0))
        self.set_enabled(True)

        linear, angular = self.tick()

        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_bypass_drives_straight_when_the_side_wall_disappears(self):
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_started = now
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.turn_direction = 1.0

        self.node.bypass_command({
            'front': math.inf,
            'left': math.inf,
            'right': math.inf,
        }, now, 0.1)

        linear, angular = self.twists.commands[-1]
        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0)

    def test_bypass_only_steers_away_from_a_close_side_wall(self):
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_started = now
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.turn_direction = 1.0

        self.node.bypass_command({
            'front': math.inf,
            'left': math.inf,
            'right': 0.20,
        }, now, 0.1)

        linear, angular = self.twists.commands[-1]
        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_avoidance_replans_after_displacement_even_if_goal_is_farther(self):
        self.prepare_plan()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        previous_plan_count = self.node.plan_count
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_active = True
        self.node.avoidance_started = now - Duration(seconds=2.0)
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.avoidance_hit_distance = 1.5
        self.node.turn_direction = 1.0
        # Goal x=1.5이므로 x=-0.10은 최종 목표에서 더 멀어진 우회다.
        self.node.current_x = -0.10

        # V3는 서로 다른 LiDAR 스캔 3회에서 헤딩/원호 안전을 확인한 뒤
        # 현재 위치에서 재계획한다.
        self.assertGreater(self.tick()[0], 0.0)
        self.node.scan_callback(make_scan())
        self.assertGreater(self.tick()[0], 0.0)
        self.node.scan_callback(make_scan())
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.plan_count, previous_plan_count + 1)
        self.assertAlmostEqual(self.node.path[0][0], -0.10)

    def test_open_planned_direction_can_rejoin_with_close_current_front(self):
        self.prepare_plan()
        now = self.node.get_clock().now()
        self.node.avoidance_started = now - Duration(seconds=2.0)
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.current_x = 0.10
        metrics = {
            'front': 0.37,
            'goal': 1.30,
            'goal_count': 5,
            'radial': 1.30,
        }

        self.node.scan_sequence = 1
        self.assertFalse(self.node.can_rejoin(metrics, now))
        self.node.scan_sequence = 2
        self.assertFalse(self.node.can_rejoin(metrics, now))
        self.node.scan_sequence = 3
        self.assertTrue(self.node.can_rejoin(metrics, now))

    def test_second_bypass_reblock_can_reselect_the_open_side(self):
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_started = now
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.turn_direction = -1.0
        self.node.avoidance_reblock_count = 1

        self.node.bypass_command({
            'front': 0.30,
            'front_left': 2.0,
            'front_right': 0.35,
            'left': 2.0,
            'right': 0.35,
            'radial': 0.50,
        }, now, 0.1)

        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)
        self.assertEqual(self.node.avoidance_direction_switches, 1)

    def test_same_scan_cannot_increment_bypass_reblock_twice(self):
        now = self.node.get_clock().now()
        self.node.avoidance_state = 'BYPASS'
        self.node.avoidance_started = now
        self.node.avoidance_start_pose = (0.0, 0.0)
        self.node.turn_direction = 1.0
        self.node.scan_sequence = 12
        metrics = {
            'front': 0.30,
            'front_left': 2.0,
            'front_right': 0.35,
            'left': 2.0,
            'right': 0.35,
            'radial': 2.0,
        }

        self.node.bypass_command(metrics, now, 0.1)
        self.assertEqual(self.node.avoidance_reblock_count, 1)

        # A second controller tick can observe the same cached LaserScan.
        # It must not turn one sensor sample into two reblock events.
        self.node.bypass_command(metrics, now, 0.1)
        self.assertEqual(self.node.avoidance_reblock_count, 1)
        self.assertEqual(
            self.node.avoidance_reblock_last_scan_sequence, 12)

    def test_blocked_corner_arc_uses_shorter_target_and_keeps_moving(self):
        blocked = [False] * (120 * 80)
        blocked[26 * 120 + 74] = True
        self.prepare_corner_tracking(blocked)

        linear, _ = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_no_safe_short_arc_stops_during_replan_cooldown(self):
        blocked = [True] * (120 * 80)
        current_cell = (62, 25)
        blocked[current_cell[1] * 120 + current_cell[0]] = False
        self.prepare_corner_tracking(blocked)

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_sharp_corner_is_not_skipped_before_reaching_it(self):
        self.node.grid_spec = GridSpec(40, 40, 0.05, -1.0, -1.0, 0.0)
        self.node.path = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.40)]
        self.node.segment_start = self.node.path[0]
        self.node.current_waypoint_index = 1
        self.node.active_goal = self.node.path[-1]
        self.node.current_x = 0.0
        self.node.current_y = 0.0

        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 1)

        self.node.current_x = 0.02
        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 1)

        self.node.current_x = 0.03
        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 2)

    def test_inflation_only_block_starts_short_escape(self):
        blocked = [True] * (120 * 80)
        current_cell = (62, 25)
        blocked[current_cell[1] * 120 + current_cell[0]] = False
        self.prepare_corner_tracking(blocked)
        self.node.raw_blocked = [False] * len(blocked)

        linear, _ = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertEqual(self.node.avoidance_state, 'BYPASS')

    def test_close_left_side_turns_right_before_inflation_escape(self):
        blocked = [True] * (120 * 80)
        current_cell = (62, 25)
        blocked[current_cell[1] * 120 + current_cell[0]] = False
        self.prepare_corner_tracking(blocked)
        self.node.raw_blocked = [False] * len(blocked)
        self.node.scan_callback(
            scan_with_front_gap(2.0, left=0.30, right=2.0))

        linear, angular = self.tick()

        self.assertEqual(linear, 0.0)
        self.assertLess(angular, 0.0)
        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, -1.0)

    def test_same_pose_arc_replan_is_not_repeated(self):
        blocked = [True] * (120 * 80)
        current_cell = (62, 25)
        blocked[current_cell[1] * 120 + current_cell[0]] = False
        self.prepare_corner_tracking(blocked)
        calls = []

        def fake_plan(reason):
            calls.append(reason)
            self.node.last_plan_time = self.node.get_clock().now()
            self.node.publish_stop()
            return True

        self.node.plan_path = fake_plan
        self.node.last_plan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.tick()
        self.assertEqual(len(calls), 1)

        self.node.last_plan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.tick()

        self.assertEqual(len(calls), 1)
        self.assertEqual(self.node.avoidance_state, 'BLOCKED')
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_intermediate_corner_is_not_skipped_at_full_tolerance(self):
        self.node.grid_spec = GridSpec(40, 40, 0.05, -1.0, -1.0, 0.0)
        self.node.path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        self.node.segment_start = self.node.path[0]
        self.node.current_waypoint_index = 1
        self.node.active_goal = self.node.path[-1]
        self.node.current_x = 0.90
        self.node.current_y = 0.0

        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 1)

        self.node.current_x = 0.98
        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 2)

    def test_dense_waypoint_accepts_one_cell_of_cross_track_error(self):
        self.node.grid_spec = GridSpec(40, 40, 0.05, -1.0, -1.0, 0.0)
        self.node.path = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        self.node.segment_start = self.node.path[0]
        self.node.current_waypoint_index = 1
        self.node.active_goal = self.node.path[-1]
        self.node.current_x = -0.058
        self.node.current_y = 0.99

        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 2)

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
            self.node.static_blocked[index] = True
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

    def test_raw_free_static_inflated_start_gets_safe_escape_prefix(self):
        message = make_map()
        # Start is cell (10, 10).  The obstacle at (10, 12) leaves the start
        # raw-free but places it inside the 0.20 m static inflation layer.
        obstacle = (10, 12)
        message.data[obstacle[1] * message.info.width + obstacle[0]] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose(0.0, 0.0))
        self.node.goal_callback(make_goal(1.5, 0.0))

        self.tick()

        start = world_to_grid(0.0, 0.0, self.node.grid_spec)
        start_index = start[1] * self.node.grid_spec.width + start[0]
        self.assertFalse(self.node.raw_blocked[start_index])
        self.assertTrue(self.node.static_blocked[start_index])
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertTrue(self.node.start_escape_active)
        self.assertTrue(self.node.path)
        self.assertEqual(self.node.path_cells[0], start)
        self.assertAlmostEqual(self.node.path[0][0], 0.0)
        self.assertAlmostEqual(self.node.path[0][1], 0.0)

        # Static inflation is relaxed only for this bounded prefix.  Raw map
        # occupancy remains authoritative and live dynamic cells stay blocked.
        escape_grid = self.node.motion_blocked_grid()
        self.assertFalse(escape_grid[start_index])
        self.node.dynamic_inflated_cells.add(start)
        self.assertTrue(self.node.motion_blocked_grid()[start_index])
        self.node.dynamic_inflated_cells.clear()

        escape_end = self.node.path_cells[self.node.start_escape_end_index]
        self.node.current_x, self.node.current_y = grid_to_world(
            escape_end[0], escape_end[1], self.node.grid_spec)
        self.assertIs(self.node.motion_blocked_grid(), self.node.blocked)
        self.assertFalse(self.node.start_escape_active)

    def test_new_goal_rearms_no_path_from_inflation_only_start(self):
        message = make_map()
        obstacle = (10, 12)
        message.data[obstacle[1] * message.info.width + obstacle[0]] = 100
        self.node.map_callback(message)
        self.node.pose_callback(make_pose(0.0, 0.0))

        # Reproduce the terminal-looking state seen after an earlier planner
        # failure, then click another goal without restarting the node.
        self.node.clear_path('test NO_PATH')
        self.node.completed = True
        self.node.set_avoidance_state('BLOCKED', 'test terminal recovery')
        self.node.goal_callback(make_goal(1.2, 0.3))
        self.tick()

        self.assertFalse(self.node.completed)
        self.assertEqual(self.node.avoidance_state, 'TRACK')
        self.assertEqual(self.node.planner_state, 'READY')
        self.assertEqual(self.node.requested_goal, (1.2, 0.3))
        self.assertTrue(self.node.start_escape_active)
        self.assertTrue(self.node.path)

    def test_large_heading_error_near_side_obstacle_blocks_spin(self):
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.0, math.pi))
        self.node.goal_callback(make_goal())
        self.tick()
        scan = make_scan()
        paint_sector(scan, 90.0, 5.0, 0.16)
        self.set_enabled(True)

        # A single close return stops one tick but no longer permanently
        # latches BLOCKED.  With no odom breadcrumbs, the confirmed third
        # sixth scan must still fail safe instead of reversing blindly.
        for _ in range(6):
            self.node.scan_callback(scan)
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

    def test_latest_pose_before_drive_enabled_is_locked_as_home(self):
        self.node.mission_active = True
        self.node.mission_home_pose = None
        self.node.mission_home_locked = False
        self.node.pose_callback(make_pose(0.0, 0.5, 0.25))

        self.assertAlmostEqual(self.node.mission_home_pose[0], 0.0)
        self.assertAlmostEqual(self.node.mission_home_pose[1], 0.5)
        self.assertAlmostEqual(self.node.mission_home_pose[2], 0.25)
        self.assertFalse(self.node.mission_home_locked)

        self.node.pose_callback(make_pose(1.0, 2.0, 0.75))
        self.assertAlmostEqual(self.node.mission_home_pose[0], 1.0)
        self.assertAlmostEqual(self.node.mission_home_pose[1], 2.0)
        self.assertAlmostEqual(self.node.mission_home_pose[2], 0.75)

        self.set_enabled(True)
        self.tick()
        self.assertTrue(self.node.mission_home_locked)

        # Later AMCL motion cannot move the recorded mission start.
        self.node.pose_callback(make_pose(2.0, 3.0, 1.0))
        self.assertAlmostEqual(self.node.mission_home_pose[0], 1.0)
        self.assertAlmostEqual(self.node.mission_home_pose[1], 2.0)
        self.assertAlmostEqual(self.node.mission_home_pose[2], 0.75)

    def test_odom_motion_updates_unlocked_home_candidate(self):
        self.node.mission_active = True
        self.node.mission_home_pose = None
        self.node.mission_home_locked = False
        self.node.odom_callback(make_odom(0.0, 0.0, 0.0))
        self.node.pose_callback(make_pose(0.0, 0.5, 0.0))
        self.node.odom_callback(make_odom(0.2, -0.1, 0.1))

        self.assertAlmostEqual(self.node.mission_home_pose[0], 0.2)
        self.assertAlmostEqual(self.node.mission_home_pose[1], 0.4)
        self.assertAlmostEqual(self.node.mission_home_pose[2], 0.1)
        self.assertFalse(self.node.mission_home_locked)

        self.set_enabled(True)
        self.tick()
        self.assertTrue(self.node.mission_home_locked)
        self.assertAlmostEqual(self.node.mission_home_pose[0], 0.2)
        self.assertAlmostEqual(self.node.mission_home_pose[1], 0.4)
        self.assertAlmostEqual(self.node.mission_home_pose[2], 0.1)

    def test_blocked_code_mission_requests_one_astar_return_home(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '테스트 주행 정지')

        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertFalse(self.node.mission_active)
        self.assertTrue(self.node.mission_failed)
        self.assertTrue(self.node.return_home_active)
        self.assertTrue(self.node.return_home_attempted)
        self.assertFalse(self.node.return_home_failed)
        self.assertEqual(self.node.goal_source, 'RETURN_HOME')
        self.assertEqual(self.node.requested_goal, (0.0, 0.5))
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_PLANNING')

        linear, angular = self.tick()
        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)
        self.assertEqual(self.node.planner_state, 'RETURNING_HOME')
        self.assertTrue(self.node.path)

    def test_return_home_arrival_latches_failed_mission_at_home(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '테스트 주행 정지')
        self.tick()
        self.tick()

        self.node.pose_callback(make_pose(0.0, 0.5))
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.completed)
        self.assertTrue(self.node.mission_failed)
        self.assertFalse(self.node.mission_completed)
        self.assertFalse(self.node.return_home_active)
        self.assertFalse(self.node.return_home_failed)
        self.assertEqual(self.node.planner_state, 'RETURNED_HOME')
        self.assertIsNone(self.node.requested_goal)
        self.assertFalse(self.node.path)

    def test_second_block_during_return_home_fails_without_restart(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '원 임무 실패')
        self.tick()
        self.tick()
        self.assertTrue(self.node.return_home_active)

        self.node.set_avoidance_state('BLOCKED', '복귀 경로도 막힘')
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.completed)
        self.assertTrue(self.node.return_home_attempted)
        self.assertFalse(self.node.return_home_active)
        self.assertTrue(self.node.return_home_failed)
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_FAILED')
        self.assertIsNone(self.node.requested_goal)

        # A later tick remains stopped and cannot recursively request HOME.
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_FAILED')

    def test_return_home_timeout_latches_failure_without_restart(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '원 임무 실패')
        self.tick()
        self.assertTrue(self.node.return_home_active)

        self.node.return_home_started = (
            self.node.get_clock().now() - Duration(seconds=361.0))
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.completed)
        self.assertFalse(self.node.return_home_active)
        self.assertTrue(self.node.return_home_failed)
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_FAILED')

        # The timeout is terminal; later ticks cannot create another return.
        plan_count = self.node.plan_count
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.plan_count, plan_count)
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_FAILED')

    def test_blocked_failure_survives_drive_disable_before_next_tick(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.tick()
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '잠금 직전 임무 실패')

        self.set_enabled(False)
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.return_home_active)
        self.assertTrue(self.node.return_home_attempted)
        self.assertEqual(self.node.goal_source, 'RETURN_HOME')
        self.assertEqual(self.node.requested_goal, (0.0, 0.5))

    def test_blocked_failure_survives_stale_pose_before_next_tick(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.tick()
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '위치 지연 직전 임무 실패')
        self.node.last_pose_time = (
            self.node.get_clock().now() - Duration(seconds=5.0))

        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.return_home_active)
        self.assertTrue(self.node.return_home_attempted)
        self.assertEqual(self.node.goal_source, 'RETURN_HOME')
        self.assertEqual(self.node.requested_goal, (0.0, 0.5))

    def test_occupied_exact_home_fails_return_instead_of_snapping(self):
        self.node.mission_waypoints = [(1.0, 0.5)]
        self.assertTrue(self.node.start_code_mission())
        self.node.map_callback(make_map())
        self.node.pose_callback(make_pose(0.0, 0.5))
        self.tick()
        self.node.scan_callback(make_scan())
        self.set_enabled(True)
        self.tick()
        self.node.pose_callback(make_pose(0.5, 0.5))
        self.node.set_avoidance_state('BLOCKED', '원 임무 실패')
        self.tick()
        self.assertTrue(self.node.return_home_active)

        home_cell = world_to_grid(0.0, 0.5, self.node.grid_spec)
        home_index = (
            home_cell[1] * self.node.grid_spec.width + home_cell[0])
        self.node.static_blocked[home_index] = True
        self.node.blocked[home_index] = True
        self.assertEqual(self.tick(), (0.0, 0.0))

        self.assertTrue(self.node.completed)
        self.assertFalse(self.node.return_home_active)
        self.assertTrue(self.node.return_home_failed)
        self.assertEqual(self.node.planner_state, 'RETURN_HOME_FAILED')
        self.assertIsNone(self.node.requested_goal)

        # A later tick remains terminal and cannot snap to another free cell.
        plan_count = self.node.plan_count
        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.plan_count, plan_count)

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
        self.assertEqual(self.node.planner_state, 'RETURNED_HOME')
        self.assertIsNone(self.node.requested_goal)
        self.assertFalse(self.node.path)
        self.assertTrue(self.node.completed)
        self.assertTrue(self.node.return_home_attempted)
        self.assertFalse(self.node.return_home_active)

        plan_count = self.node.plan_count
        changed_map = make_map()
        changed_map.data[35 * changed_map.info.width + 35] = 100
        self.node.map_callback(changed_map)
        self.tick()
        self.assertEqual(self.node.plan_count, plan_count)
        self.assertEqual(self.node.planner_state, 'RETURNED_HOME')

    def test_spin_block_unwinds_to_last_forward_heading_before_backtrack(self):
        metrics = {
            'radial': 0.10,
            'rear': 1.0,
            'rear_range_count': 20,
        }
        self.prepare_backtrack_case()
        result = self.node.set_parameters([
            Parameter(
                'spin_block_confirm_scans',
                Parameter.Type.INTEGER,
                1,
            ),
        ])[0]
        self.assertTrue(result.successful, result.reason)

        # Simulate a half-radian in-place turn after the final forward
        # breadcrumb.  That yaw-only odom sample must not replace the saved
        # forward heading.
        self.node.odom_callback(make_odom(-0.25, 0.0, math.pi + 0.5))
        self.node.scan_sequence += 1
        now = self.node.get_clock().now()

        self.assertTrue(self.node.handle_spin_block(metrics, now))
        self.assertEqual(self.node.avoidance_state, 'UNWIND_TURN')
        self.assertAlmostEqual(
            self.node.unwind_target_yaw,
            self.node.forward_trail[-1][2],
        )
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

        # Once odom reports the saved heading again, UNWIND_TURN hands over to
        # the existing breadcrumb reverse controller instead of declaring the
        # trail invalid merely because the robot had rotated.
        self.node.odom_callback(make_odom(-0.25, 0.0, math.pi))
        now = self.node.get_clock().now()
        self.node.unwind_turn_command(metrics, now, 0.1)

        self.assertEqual(self.node.avoidance_state, 'BACKTRACK')
        self.assertGreaterEqual(len(self.node.backtrack_path), 2)
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))

    def test_gazebo_time_reset_clears_future_cooldown_and_old_path(self):
        self.prepare_plan()
        now = self.node.get_clock().now()
        self.node.dynamic_expiry[(12, 12)] = now.nanoseconds + 5_000_000_000
        self.node._compose_blocked_grid()
        self.node.last_dynamic_clock_ns = now.nanoseconds + 2_000_000_000
        self.node.last_dynamic_replan_time = now + Duration(seconds=10.0)

        self.assertTrue(self.node._expire_dynamic_obstacles(now))

        self.assertFalse(self.node.dynamic_expiry)
        self.assertIsNone(self.node.last_dynamic_replan_time)
        self.assertFalse(self.node.is_localized)
        self.assertFalse(self.node.path)
        self.assertIsNone(self.node.active_goal)
        self.assertEqual(self.node.planner_state, 'WAIT_POSE')
        self.assertTrue(self.node.replan_requested)
        self.assertEqual(self.twists.commands[-1], (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
