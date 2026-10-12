import math
import unittest

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan

from my_turtle_pkg.pure_pursuit_obstacle import (
    PurePursuitLogic,
    WAFFLE_PI_FRONT_X,
    WAFFLE_PI_LIDAR_X,
    front_body_clearance,
    scan_sector_distance,
    select_lookahead_point,
)


def make_scan(
        angle_min=-math.pi,
        angle_increment=math.pi / 180.0,
        count=361,
        default=math.inf):
    scan = LaserScan()
    scan.angle_min = float(angle_min)
    scan.angle_increment = float(angle_increment)
    scan.angle_max = scan.angle_min + (count - 1) * scan.angle_increment
    scan.range_min = 0.12
    scan.range_max = 10.0
    scan.ranges = [default] * count
    return scan


def paint_sector(scan, center_degrees, half_width_degrees, distance):
    center = math.radians(center_degrees)
    half_width = math.radians(half_width_degrees)
    for index in range(len(scan.ranges)):
        angle = scan.angle_min + index * scan.angle_increment
        error = (angle - center + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error) <= half_width:
            scan.ranges[index] = distance


def scan_with_front_gap(gap, left=2.0, right=0.55):
    scan = make_scan()
    raw_range = gap + WAFFLE_PI_FRONT_X - WAFFLE_PI_LIDAR_X
    paint_sector(scan, 0.0, 7.0, raw_range)
    paint_sector(scan, 45.0, 12.0, left)
    paint_sector(scan, 90.0, 20.0, left)
    paint_sector(scan, -45.0, 12.0, right)
    paint_sector(scan, -90.0, 20.0, right)
    return scan


def make_pose(x=0.0, y=0.0, yaw=0.0):
    pose = PoseWithCovarianceStamped()
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
    def test_front_sector_uses_laserscan_angles_in_both_directions(self):
        for scan in (
                make_scan(-math.pi, math.pi / 180.0),
                make_scan(math.pi, -math.pi / 180.0)):
            paint_sector(scan, 0.0, 3.0, 0.72)
            distance, count = scan_sector_distance(
                scan, 0.0, math.radians(5.0), percentile=0.05)
            self.assertGreaterEqual(count, 3)
            self.assertAlmostEqual(distance, 0.72)

    def test_body_clearance_uses_multiple_returns_not_one_outlier(self):
        scan = make_scan()
        center = len(scan.ranges) // 2
        raw_range = 0.30 + WAFFLE_PI_FRONT_X - WAFFLE_PI_LIDAR_X

        scan.ranges[center] = raw_range
        self.assertTrue(math.isinf(
            front_body_clearance(scan, half_width=0.18, min_hits=3)))

        for index in range(center - 1, center + 2):
            scan.ranges[index] = raw_range
        self.assertAlmostEqual(
            front_body_clearance(scan, half_width=0.18, min_hits=3),
            0.30,
            delta=0.001,
        )

    def test_obstacle_inside_body_front_is_zero_clearance(self):
        scan = make_scan()
        center = len(scan.ranges) // 2
        for index in range(center - 1, center + 2):
            scan.ranges[index] = scan.range_min

        self.assertEqual(
            front_body_clearance(scan, half_width=0.18, min_hits=3),
            0.0,
        )

    def test_lookahead_follows_polyline_around_corner(self):
        target = select_lookahead_point(
            x=0.8,
            y=0.0,
            segment_start=(0.0, 0.0),
            path=[(1.0, 0.0), (1.0, 1.0)],
            waypoint_index=0,
            lookahead_distance=0.5,
        )
        self.assertAlmostEqual(target[0], 1.0)
        self.assertAlmostEqual(target[1], 0.3)

    def test_lookahead_returns_to_missed_waypoint_after_pass_plane(self):
        target = select_lookahead_point(
            x=1.2,
            y=0.8,
            segment_start=(0.0, 0.0),
            path=[(1.0, 0.0), (1.0, 1.0)],
            waypoint_index=0,
            lookahead_distance=0.5,
        )
        self.assertEqual(target, (1.0, 0.0))


class PurePursuitObstacleNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = PurePursuitLogic()
        self.node.timer.cancel()
        self.capture = CapturePublisher()
        self.node.publisher_ = self.capture
        self.node.path = [(2.0, 0.0)]
        self.node.pose_callback(make_pose())

    def tearDown(self):
        self.node.destroy_node()

    def set_drive_enabled(self, enabled):
        result = self.node.set_parameters([
            Parameter('drive_enabled', Parameter.Type.BOOL, enabled),
        ])[0]
        self.assertTrue(result.successful, result.reason)

    def tick(self):
        self.node.last_control_time = (
            self.node.get_clock().now() - Duration(seconds=0.1))
        self.node.control_loop()
        return self.capture.commands[-1]

    def test_open_scan_tracks_path(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())

        linear, angular = self.tick()

        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0, delta=0.02)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_all_nan_scan_is_invalid_and_hard_stops(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan(default=math.nan))

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'WAIT_SCAN')

    def test_out_of_range_scan_is_invalid_and_hard_stops(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan(default=100.0))

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'WAIT_SCAN')

    def test_invalid_pose_does_not_unlock_motion(self):
        invalid_pose = make_pose()
        invalid_pose.pose.pose.position.x = math.nan
        self.node.is_localized = False
        self.node.last_pose_time = None
        self.node.pose_callback(invalid_pose)
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertFalse(self.node.is_localized)

    def test_enabling_does_not_refresh_a_stale_pose(self):
        self.node.last_pose_time = (
            self.node.get_clock().now() - Duration(seconds=5.0))
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertEqual(self.node.avoidance_state, 'WAIT_POSE')

    def test_finished_mission_stops_before_obstacle_avoidance(self):
        self.node.path = [(0.0, 0.0)]
        self.set_drive_enabled(True)
        self.node.scan_callback(scan_with_front_gap(0.25))

        self.assertEqual(self.tick(), (0.0, 0.0))
        self.assertTrue(self.node.completed)
        self.assertEqual(self.node.avoidance_state, 'TRACK')

    def test_turn_direction_stays_latched_while_obstacle_remains(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(
            scan_with_front_gap(0.30, left=2.0, right=0.55))
        _, first_angular = self.tick()

        self.assertEqual(self.node.avoidance_state, 'TURN')
        self.assertEqual(self.node.turn_direction, 1.0)
        self.assertGreater(first_angular, 0.0)

        self.node.scan_callback(
            scan_with_front_gap(0.30, left=0.55, right=2.0))
        _, second_angular = self.tick()

        self.assertEqual(self.node.turn_direction, 1.0)
        self.assertGreater(second_angular, 0.0)

    def test_stale_scan_stops_both_axes(self):
        self.set_drive_enabled(True)
        self.node.scan_callback(make_scan())
        self.assertGreater(self.tick()[0], 0.0)

        self.node.last_scan_time = (
            self.node.get_clock().now() - Duration(seconds=2.0))
        self.assertEqual(self.tick(), (0.0, 0.0))

    def test_multiple_reached_waypoints_advance_in_one_tick(self):
        self.node.path = [(0.05, 0.0), (0.10, 0.0), (1.0, 0.0)]
        self.node.segment_start = (0.0, 0.0)

        self.assertFalse(self.node.advance_waypoints())
        self.assertEqual(self.node.current_waypoint_index, 2)

    def test_invalid_live_parameter_update_is_rejected(self):
        result = self.node.set_parameters([
            Parameter('lookahead_distance', Parameter.Type.DOUBLE, 0.0),
        ])[0]
        self.assertFalse(result.successful)


if __name__ == '__main__':
    unittest.main()
