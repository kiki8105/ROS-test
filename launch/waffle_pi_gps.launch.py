"""Waffle Pi GPS 모델, 고장 주입기, hybrid 위치원을 한 번에 실행한다."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('my_turtle_pkg')
    gazebo_share = get_package_share_directory('gazebo_ros')
    tb3_share = get_package_share_directory('turtlebot3_gazebo')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    gps_mode = LaunchConfiguration('gps_mode')
    run_pursuit = LaunchConfiguration('run_pursuit')
    run_drive_stack = LaunchConfiguration('run_drive_stack')
    operation_mode = LaunchConfiguration('operation_mode')

    model_file = os.path.join(
        package_share, 'models', 'turtlebot3_waffle_pi_gps', 'model.sdf')
    urdf_file = os.path.join(tb3_share, 'urdf', 'turtlebot3_waffle_pi.urdf')
    with open(urdf_file, encoding='utf-8') as stream:
        robot_description = stream.read()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': gui}.items())

    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}])

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'waffle_pi_gps', '-file', model_file,
            '-x', x, '-y', y, '-z', '0.01', '-Y', yaw])

    sensor_params = os.path.join(package_share, 'config', 'hybrid_localizer.yaml')
    fault_injector = Node(
        package='my_turtle_pkg',
        executable='gps_fault_injector',
        name='gps_fault_injector',
        output='screen',
        parameters=[
            sensor_params,
            {'mode': ParameterValue(gps_mode, value_type=str), 'use_sim_time': True},
        ])
    localizer = Node(
        package='my_turtle_pkg',
        executable='hybrid_localizer',
        name='hybrid_localizer',
        output='screen',
        parameters=[sensor_params, {'use_sim_time': True}])
    pursuit = Node(
        package='my_turtle_pkg',
        executable='pure_pursuit_v3',
        name='pure_pursuit_v3_node',
        output='screen',
        condition=IfCondition(run_pursuit),
        parameters=[
            os.path.join(package_share, 'config', 'pure_pursuit_gps_v3.yaml'),
            {'use_sim_time': True},
        ])
    vision = Node(
        package='my_turtle_pkg',
        executable='red_line_follower',
        name='red_line_follower',
        output='screen',
        condition=IfCondition(run_drive_stack),
        parameters=[
            os.path.join(package_share, 'config', 'red_line.yaml'),
            {
                'cmd_vel_topic': '/cmd_vel/vision',
                'line_valid_topic': '/vision/line_valid',
                'show_debug': False,
            },
        ])
    manager = Node(
        package='my_turtle_pkg',
        executable='drive_mode_manager',
        name='drive_mode_manager',
        output='screen',
        condition=IfCondition(run_drive_stack),
        parameters=[
            os.path.join(package_share, 'config', 'drive_mode_manager.yaml'),
            {'operation_mode': ParameterValue(operation_mode, value_type=str)},
        ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=os.path.join(package_share, 'worlds', 'gps_line_test.world')),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x', default_value='1.5'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='2.09439510239'),
        DeclareLaunchArgument(
            'gps_mode', default_value='normal',
            description='normal|dropout|freeze|degraded|no_fix'),
        DeclareLaunchArgument(
            'run_pursuit', default_value='true',
            description='true면 /hybrid_pose를 사용하는 Pure Pursuit v3도 실행'),
        DeclareLaunchArgument(
            'run_drive_stack', default_value='true',
            description='GPS/vision 명령 선택기와 차선 추종 노드 실행'),
        DeclareLaunchArgument(
            'operation_mode', default_value='outdoor',
            description='outdoor|indoor|auto'),
        gazebo,
        state_publisher,
        spawn,
        fault_injector,
        localizer,
        pursuit,
        vision,
        manager,
    ])
