import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('volleyball_catch_controller')
    catch_config = os.path.join(share, 'config', 'catch_controller_d435_only.yaml')
    velocity_config = os.path.join(share, 'config', 'goal_to_cmd_vel.yaml')
    odom_topic = LaunchConfiguration('odom_topic')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/vision/cmd_vel'),
        Node(
            package='volleyball_catch_controller',
            executable='catch_controller',
            name='catch_controller',
            output='screen',
            parameters=[catch_config, {'odom_topic': odom_topic}],
        ),
        Node(
            package='volleyball_catch_controller',
            executable='goal_to_cmd_vel',
            name='goal_to_cmd_vel',
            output='screen',
            parameters=[velocity_config, {'cmd_vel_topic': cmd_vel_topic}],
        ),
    ])
