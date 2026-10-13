"""Saved-map localization + integrated A* / Pure Pursuit V3."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('my_turtle_pkg')
    nav2_share = get_package_share_directory('nav2_bringup')

    map_file = LaunchConfiguration('map')
    nav2_params = LaunchConfiguration('nav2_params')
    pursuit_params = LaunchConfiguration('pure_pursuit_params')
    use_sim_time = LaunchConfiguration('use_sim_time')
    localization_autostart = LaunchConfiguration('localization_autostart')

    # Nav2의 planner/controller는 시작하지 않습니다. V3가 A*와
    # /cmd_vel을 모두 담당하므로 자율주행 명령 발행자는 하나뿐입니다.
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, 'launch', 'localization_launch.py')),
        launch_arguments={
            'map': map_file,
            'params_file': nav2_params,
            'use_sim_time': use_sim_time,
            'autostart': localization_autostart,
            'use_composition': 'False',
            'use_respawn': 'False',
        }.items(),
    )

    pursuit = Node(
        package='my_turtle_pkg',
        executable='pure_pursuit_obstacle_v3',
        name='pure_pursuit_obstacle_v3',
        output='screen',
        emulate_tty=True,
        parameters=[
            pursuit_params,
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=(
                '/home/kiki/turtlebot3/maps/'
                'patrol_20260930_133038/map.yaml'),
            description='저장 지도 YAML 경로'),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=(
                '/home/kiki/turtlebot3/install/turtlebot3_navigation2/'
                'share/turtlebot3_navigation2/param/humble/waffle_pi.yaml'),
            description='map_server와 AMCL용 Nav2 파라미터 파일'),
        DeclareLaunchArgument(
            'pure_pursuit_params',
            default_value=os.path.join(
                package_share, 'config', 'pure_pursuit_obstacle_v3.yaml'),
            description='V3 A*, Pure Pursuit, LiDAR 회피 파라미터'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Gazebo에서는 true, 실차에서는 false'),
        DeclareLaunchArgument(
            'localization_autostart', default_value='true',
            description='map_server와 AMCL lifecycle 자동 시작'),
        localization,
        pursuit,
    ])
