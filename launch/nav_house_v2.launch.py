"""House 저장 지도 localization + 교육용 Pure Pursuit 주행 노드."""
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
        }.items())

    pursuit = Node(
        package='my_turtle_pkg',
        executable='pure_pursuit',
        name='pure_pursuit_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            pursuit_params,
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
        ])

    lifecycle_guard = Node(
        package='my_turtle_pkg',
        executable='localization_guard',
        name='localization_startup_guard',
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='/home/kiki/turtlebot3/maps/patrol_20260930_133038/map.yaml',
            description='House 저장 지도 YAML 경로'),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=(
                '/home/kiki/turtlebot3/install/turtlebot3_navigation2/'
                'share/turtlebot3_navigation2/param/humble/waffle_pi.yaml'),
            description='map_server와 AMCL이 사용할 Nav2 파라미터 파일'),
        DeclareLaunchArgument(
            'pure_pursuit_params',
            default_value=os.path.join(package_share, 'config', 'pure_pursuit_house.yaml'),
            description='Pure Pursuit 속도·가속도·경유점 파라미터 파일'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Gazebo /clock 사용 여부'),
        DeclareLaunchArgument(
            'localization_autostart', default_value='true',
            description='표준 lifecycle manager 자동 시작 사용 여부'),
        localization,
        lifecycle_guard,
        pursuit,
    ])
