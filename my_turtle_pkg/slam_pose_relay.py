"""Cartographer의 map->base TF를 /slam_pose와 health 토픽으로 변환한다."""

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener


class SlamPoseRelay(Node):
    def __init__(self):
        super().__init__('slam_pose_relay')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('output_topic', '/slam_pose')
        self.declare_parameter('health_topic', '/slam/healthy')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('transform_timeout', 0.7)
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter('output_topic').value), 10)
        self.health_pub = self.create_publisher(
            Bool, str(self.get_parameter('health_topic').value), 10)
        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(1.0 / rate, self.update)

    def update(self):
        try:
            transform = self.buffer.lookup_transform(
                str(self.get_parameter('map_frame').value),
                str(self.get_parameter('base_frame').value), Time())
        except TransformException:
            self.health_pub.publish(Bool(data=False))
            return
        stamp = transform.header.stamp
        transform_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter('transform_timeout').value)
        if transform_time <= 0.0 or now - transform_time > timeout:
            self.health_pub.publish(Bool(data=False))
            return
        msg = PoseWithCovarianceStamped()
        msg.header = transform.header
        msg.pose.pose.position.x = transform.transform.translation.x
        msg.pose.pose.position.y = transform.transform.translation.y
        msg.pose.pose.position.z = transform.transform.translation.z
        msg.pose.pose.orientation = transform.transform.rotation
        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.03
        self.pose_pub.publish(msg)
        self.health_pub.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    node = SlamPoseRelay()
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
