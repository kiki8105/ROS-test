"""정육각형 빨간 선 + Waffle Pi + 비전 추종. 기본 시작 위치 (1.5, 0), 방향 120도."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('my_turtle_pkg')
    gazebo = get_package_share_directory('gazebo_ros')
    tb = get_package_share_directory('turtlebot3_gazebo')
    with open(os.path.join(tb, 'urdf', 'turtlebot3_waffle_pi.urdf')) as f:
        description = f.read()
    spawner = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                   arguments=['-entity', 'red_line_waffle_pi', '-file',
                              os.path.join(tb, 'models', 'turtlebot3_waffle_pi', 'model.sdf'),
                              '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                              '-z', '0.01', '-Y', LaunchConfiguration('yaw')])
    follower = Node(package='my_turtle_pkg', executable='red_line_follower', output='screen',
                    condition=IfCondition(LaunchConfiguration('follow')),
                    parameters=[os.path.join(pkg, 'config', 'red_line.yaml'),
                                {'show_debug': ParameterValue(LaunchConfiguration('show_debug'), value_type=bool)}])

    def after_spawn(event, context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(reason='터틀봇 생성 실패'))]
        return [follower]

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='1.5'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='2.09439510239'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('follow', default_value='true'),
        DeclareLaunchArgument('show_debug', default_value='true'),
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds', 'red_line_empty.world')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo, 'launch', 'gazebo.launch.py')),
                                 launch_arguments={'world': LaunchConfiguration('world'),
                                                   'gui': LaunchConfiguration('gui')}.items()),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description, 'use_sim_time': True}]),
        RegisterEventHandler(OnProcessExit(target_action=spawner, on_exit=after_spawn)),
        spawner,
    ])
