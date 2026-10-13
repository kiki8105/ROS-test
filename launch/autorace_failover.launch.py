"""GPS fault injector, vision, SLAM pose, mission controller, safety mux."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('my_turtle_pkg')
    hybrid = os.path.join(package_share, 'config', 'autorace_hybrid.yaml')
    lane = os.path.join(package_share, 'config', 'autorace_lane.yaml')
    mission = os.path.join(package_share, 'config', 'autorace_failover_mission.yaml')
    mission_enabled = LaunchConfiguration('mission_enabled')
    gps_mode = LaunchConfiguration('gps_mode')
    return LaunchDescription([
        DeclareLaunchArgument('mission_enabled', default_value='false'),
        DeclareLaunchArgument(
            'gps_mode', default_value='normal',
            description='normal|dropout|freeze|degraded|no_fix'),
        Node(
            package='my_turtle_pkg', executable='gps_fault_injector',
            name='gps_fault_injector', output='screen',
            parameters=[hybrid, {
                'mode': ParameterValue(gps_mode, value_type=str),
                'use_sim_time': True,
            }]),
        Node(
            package='my_turtle_pkg', executable='slam_pose_relay',
            name='slam_pose_relay', output='screen', parameters=[hybrid]),
        Node(
            package='my_turtle_pkg', executable='hybrid_localizer',
            name='hybrid_localizer', output='screen', parameters=[hybrid]),
        Node(
            package='my_turtle_pkg', executable='autorace_lane_follower',
            name='autorace_lane_follower', output='screen', parameters=[lane]),
        Node(
            package='my_turtle_pkg', executable='autorace_failover_mission',
            name='autorace_failover_mission', output='screen',
            parameters=[mission, {
                'mission_enabled': ParameterValue(mission_enabled, value_type=bool),
            }]),
        Node(
            package='my_turtle_pkg', executable='safety_cmd_mux',
            name='safety_cmd_mux', output='screen', parameters=[mission]),
    ])
