"""Recover a Nav2 localization startup that stalls during lifecycle transitions."""
import time

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node


class LocalizationGuard(Node):
    """Wait for Nav2, then repair only incomplete localization startup states."""

    def __init__(self):
        super().__init__('localization_startup_guard')
        self.declare_parameter('startup_grace_sec', 5.0)
        self.declare_parameter('startup_timeout_sec', 60.0)

        self.started = time.monotonic()
        self.pending = False
        self.finished = False
        self.success = False
        self.state_clients = {
            name: self.create_client(GetState, f'/{name}/get_state')
            for name in ('map_server', 'amcl')
        }
        self.change_clients = {
            name: self.create_client(ChangeState, f'/{name}/change_state')
            for name in ('map_server', 'amcl')
        }
        self.timer = self.create_timer(0.5, self.check_startup)

    def check_startup(self):
        if self.pending or self.finished:
            return
        elapsed = time.monotonic() - self.started
        grace = float(self.get_parameter('startup_grace_sec').value)
        timeout = float(self.get_parameter('startup_timeout_sec').value)
        if elapsed < grace:
            return
        if elapsed > timeout:
            self.get_logger().error(
                'Localization lifecycle 활성화 제한 시간을 초과했습니다. '
                'map_server/amcl 로그를 확인하세요.')
            self.finished = True
            return
        self.request_state('map_server')

    def request_state(self, name):
        client = self.state_clients[name]
        if not client.service_is_ready():
            self.get_logger().warn(
                f'/{name}/get_state 서비스 대기 중...', throttle_duration_sec=3.0)
            return
        self.pending = True
        future = client.call_async(GetState.Request())
        future.add_done_callback(lambda result, node=name: self.state_received(node, result))

    def state_received(self, name, future):
        self.pending = False
        try:
            state = future.result().current_state.id
        except Exception as exc:
            self.get_logger().warn(f'{name} 상태 조회 실패, 재시도합니다: {exc}')
            return

        if state == State.PRIMARY_STATE_ACTIVE:
            if name == 'map_server':
                self.request_state('amcl')
            else:
                self.success = True
                self.finished = True
                self.get_logger().info('map_server와 amcl이 모두 active입니다.')
            return
        if state == State.PRIMARY_STATE_UNCONFIGURED:
            self.request_transition(name, Transition.TRANSITION_CONFIGURE, 'configure')
        elif state == State.PRIMARY_STATE_INACTIVE:
            self.request_transition(name, Transition.TRANSITION_ACTIVATE, 'activate')
        else:
            self.get_logger().warn(
                f'{name}이 전환 상태({state})입니다. 완료될 때까지 기다립니다.',
                throttle_duration_sec=2.0)

    def request_transition(self, name, transition_id, label):
        client = self.change_clients[name]
        if not client.service_is_ready():
            self.get_logger().warn(
                f'/{name}/change_state 서비스 대기 중...', throttle_duration_sec=3.0)
            return
        request = ChangeState.Request()
        request.transition.id = transition_id
        self.pending = True
        self.get_logger().warn(
            f'lifecycle 자동 시작이 완료되지 않아 {name} {label}를 재시도합니다.')
        future = client.call_async(request)
        future.add_done_callback(
            lambda result, node=name, action=label:
            self.transition_finished(node, action, result))

    def transition_finished(self, name, label, future):
        self.pending = False
        try:
            successful = future.result().success
        except Exception as exc:
            self.get_logger().warn(
                f'{name} {label} 응답 실패, 상태를 다시 확인합니다: {exc}')
            return
        if successful:
            self.get_logger().info(f'{name} {label} 완료')
        else:
            self.get_logger().warn(
                f'{name} {label}가 거부됐습니다. 현재 상태를 다시 확인합니다.')


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationGuard()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
