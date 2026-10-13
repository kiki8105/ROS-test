"""Gazebo GPS에 재현 가능한 고장을 주입하는 ROS 2 노드."""

import copy
import math
import random

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String
from std_srvs.srv import SetBool


VALID_MODES = ('normal', 'dropout', 'freeze', 'degraded', 'no_fix')


def add_horizontal_noise(fix, east_m, north_m, stddev_m):
    """NavSatFix 복사본에 ENU 미터 오차를 위도/경도로 환산해 더한다."""
    result = copy.deepcopy(fix)
    meters_per_degree = 111_320.0
    longitude_scale = max(
        meters_per_degree * math.cos(math.radians(fix.latitude)), 1.0)
    result.latitude += north_m / meters_per_degree
    result.longitude += east_m / longitude_scale
    variance = float(stddev_m) ** 2
    result.position_covariance[0] = variance
    result.position_covariance[4] = variance
    result.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
    return result


class GpsFaultInjector(Node):
    def __init__(self):
        super().__init__('gps_fault_injector')
        self.declare_parameter('input_topic', '/fix/raw')
        self.declare_parameter('output_topic', '/fix')
        self.declare_parameter('mode', 'normal')
        self.declare_parameter('degraded_stddev_m', 5.0)
        self.declare_parameter('random_seed', 64)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.mode = str(self.get_parameter('mode').value)
        if self.mode not in VALID_MODES:
            raise ValueError(f'mode는 {VALID_MODES} 중 하나여야 합니다.')

        self.random = random.Random(int(self.get_parameter('random_seed').value))
        self.frozen_fix = None
        self.fix_pub = self.create_publisher(NavSatFix, output_topic, qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, '/gps/fault_mode', 10)
        self.create_subscription(
            NavSatFix, input_topic, self.fix_callback, qos_profile_sensor_data)
        self.create_service(SetBool, '/gps_failure/set', self.failure_service)
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info(
            f'GPS 고장 주입기: {input_topic} -> {output_topic}, mode={self.mode}')

    def set_mode(self, mode):
        if mode not in VALID_MODES:
            return False
        if mode != self.mode:
            self.mode = mode
            self.frozen_fix = None
            self.get_logger().warn(f'GPS failure mode -> {mode}')
            self.publish_status()
        return True

    def parameter_callback(self, parameters):
        for parameter in parameters:
            if parameter.name == 'mode' and not self.set_mode(str(parameter.value)):
                return SetParametersResult(
                    successful=False,
                    reason=f'mode는 {VALID_MODES} 중 하나여야 합니다.')
            if parameter.name == 'degraded_stddev_m' and float(parameter.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason='degraded_stddev_m은 0 이상이어야 합니다.')
        return SetParametersResult(successful=True)

    def failure_service(self, request, response):
        self.set_mode('dropout' if request.data else 'normal')
        response.success = True
        response.message = f'GPS mode={self.mode}'
        return response

    def publish_status(self):
        self.status_pub.publish(String(data=self.mode))

    def fix_callback(self, msg):
        if self.mode == 'dropout':
            return
        if self.mode == 'freeze':
            if self.frozen_fix is None:
                self.frozen_fix = copy.deepcopy(msg)
            self.fix_pub.publish(self.frozen_fix)
            return
        if self.mode == 'no_fix':
            failed = copy.deepcopy(msg)
            failed.status.status = NavSatStatus.STATUS_NO_FIX
            failed.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            self.fix_pub.publish(failed)
            return
        if self.mode == 'degraded':
            stddev = float(self.get_parameter('degraded_stddev_m').value)
            degraded = add_horizontal_noise(
                msg,
                self.random.gauss(0.0, stddev),
                self.random.gauss(0.0, stddev),
                stddev)
            self.fix_pub.publish(degraded)
            return
        self.fix_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GpsFaultInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
