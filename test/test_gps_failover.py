import math
import unittest

from sensor_msgs.msg import NavSatFix

from my_turtle_pkg.gps_fault_injector import add_horizontal_noise
from my_turtle_pkg.hybrid_localizer import (
    apply_se2_transform,
    estimate_se2_transform,
    gps_to_local,
    propagate_pose,
    slam_pose_in_map,
    transform_local,
)
from my_turtle_pkg.drive_mode_manager import choose_source


class GpsFailoverMathTest(unittest.TestCase):
    def test_gps_to_local_one_meter_east(self):
        lat0 = 37.5665
        lon0 = 126.9780
        longitude = lon0 + 1.0 / (111_320.0 * math.cos(math.radians(lat0)))
        east, north = gps_to_local(lat0, longitude, lat0, lon0)
        self.assertAlmostEqual(east, 1.0, places=3)
        self.assertAlmostEqual(north, 0.0, places=6)

    def test_map_alignment_rotation(self):
        x, y = transform_local(1.0, 0.0, 5.0, 7.0, math.pi / 2.0)
        self.assertAlmostEqual(x, 5.0, places=6)
        self.assertAlmostEqual(y, 8.0, places=6)

    def test_odom_fallback_propagates_in_map_frame(self):
        pose = propagate_pose(
            (5.0, 10.0, math.pi / 2.0),
            (2.0, 3.0, 0.0),
            (3.0, 3.0, math.pi / 4.0))
        self.assertAlmostEqual(pose[0], 5.0, places=6)
        self.assertAlmostEqual(pose[1], 11.0, places=6)
        self.assertAlmostEqual(pose[2], 3.0 * math.pi / 4.0, places=6)

    def test_slam_pose_is_aligned_to_gps_world_frame(self):
        slam_at_fix = (0.0, 0.0, 0.0)
        gps_at_fix = (0.8, -1.747, 0.0)
        transform = estimate_se2_transform(slam_at_fix, gps_at_fix)

        aligned_at_fix = apply_se2_transform(slam_at_fix, transform)
        self.assertAlmostEqual(aligned_at_fix[0], gps_at_fix[0], places=6)
        self.assertAlmostEqual(aligned_at_fix[1], gps_at_fix[1], places=6)
        self.assertAlmostEqual(aligned_at_fix[2], gps_at_fix[2], places=6)

        moved = slam_pose_in_map((0.5, 0.2, 0.1), transform)
        self.assertAlmostEqual(moved[0], 1.3, places=6)
        self.assertAlmostEqual(moved[1], -1.547, places=6)
        self.assertAlmostEqual(moved[2], 0.1, places=6)

    def test_slam_alignment_supports_rotated_frames(self):
        slam_at_fix = (1.0, 2.0, math.pi / 2.0)
        gps_at_fix = (5.0, 7.0, math.pi)
        transform = estimate_se2_transform(slam_at_fix, gps_at_fix)
        aligned = apply_se2_transform(slam_at_fix, transform)
        self.assertAlmostEqual(aligned[0], gps_at_fix[0], places=6)
        self.assertAlmostEqual(aligned[1], gps_at_fix[1], places=6)
        self.assertAlmostEqual(aligned[2], gps_at_fix[2], places=6)
        moved = apply_se2_transform((2.0, 2.0, math.pi / 2.0), transform)
        self.assertAlmostEqual(moved[0], 5.0, places=6)
        self.assertAlmostEqual(moved[1], 8.0, places=6)
        self.assertAlmostEqual(moved[2], math.pi, places=6)

    def test_slam_without_prior_gps_keeps_its_native_frame(self):
        pose = (1.2, -0.4, 0.3)
        self.assertEqual(slam_pose_in_map(pose, None), pose)

    def test_degraded_fix_changes_covariance(self):
        fix = NavSatFix()
        fix.latitude = 37.5665
        fix.longitude = 126.9780
        changed = add_horizontal_noise(fix, 2.0, -3.0, 5.0)
        self.assertNotEqual(changed.latitude, fix.latitude)
        self.assertNotEqual(changed.longitude, fix.longitude)
        self.assertEqual(changed.position_covariance[0], 25.0)
        self.assertEqual(changed.position_covariance[4], 25.0)

    def test_outdoor_gps_failure_changes_to_vision(self):
        self.assertEqual(
            choose_source('outdoor', False, True, True, True), 'vision')

    def test_outdoor_without_lane_stops_safely(self):
        self.assertEqual(
            choose_source('outdoor', False, False, True, True), 'stop')

    def test_indoor_always_uses_valid_lane(self):
        self.assertEqual(
            choose_source('indoor', True, True, True, True), 'vision')


if __name__ == '__main__':
    unittest.main()
