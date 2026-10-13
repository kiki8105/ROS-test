import math
import unittest

from my_turtle_pkg.pure_pursuit import (
    normalize_angle,
    pursuit_command,
    rate_limit,
)


class PurePursuitMathTests(unittest.TestCase):
    def test_normalize_angle_wraps_to_signed_pi_range(self):
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(normalize_angle(-1.5 * math.pi), 0.5 * math.pi)

    def test_pursuit_command_uses_heading_error_and_angular_limit(self):
        distance, alpha, linear, angular = pursuit_command(
            x=0.0,
            y=0.0,
            yaw=0.0,
            goal_x=1.0,
            goal_y=1.0,
            linear_velocity=0.2,
            lookahead_distance=0.5,
            max_angular_velocity=0.3,
        )

        self.assertAlmostEqual(distance, math.sqrt(2.0))
        self.assertAlmostEqual(alpha, math.pi / 4.0)
        self.assertAlmostEqual(linear, 0.2)
        self.assertAlmostEqual(angular, 0.3)

        _, right_alpha, _, right_angular = pursuit_command(
            0.0, 0.0, 0.0, 1.0, -1.0, 0.2, 0.5, 1.0)
        self.assertLess(right_alpha, 0.0)
        self.assertLess(right_angular, 0.0)

    def test_rate_limit_bounds_acceleration_and_deceleration_per_cycle(self):
        self.assertAlmostEqual(rate_limit(0.0, 1.0, limit=0.2, dt=0.1), 0.02)
        self.assertAlmostEqual(rate_limit(0.5, 0.0, limit=0.4, dt=0.25), 0.4)
        self.assertAlmostEqual(rate_limit(-0.5, 0.5, limit=2.0, dt=0.1), -0.3)


if __name__ == '__main__':
    unittest.main()
