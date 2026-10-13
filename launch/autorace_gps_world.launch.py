"""Autorace 2020 월드에 GPS Waffle Pi만 생성한다."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('my_turtle_pkg')
    gazebo_share = get_package_share_directory('gazebo_ros')
    tb3_share = get_package_share_directory('turtlebot3_gazebo')
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    with open(os.path.join(tb3_share, 'urdf', 'turtlebot3_waffle_pi.urdf'),
              encoding='utf-8') as stream:
        robot_description = stream.read()
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=os.path.join(
                package_share, 'worlds', 'autorace_gps_2020.world')),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x', default_value='0.8'),
        DeclareLaunchArgument('y', default_value='-1.747'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': world, 'gui': gui}.items()),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen'),
        Node(
            package='gazebo_ros', executable='spawn_entity.py', output='screen',
            arguments=[
                '-entity', 'waffle_pi_gps_autorace',
                '-file', os.path.join(
                    package_share, 'models', 'turtlebot3_waffle_pi_gps', 'model.sdf'),
                '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                '-z', '0.01', '-Y', LaunchConfiguration('yaw')]),
    ])
