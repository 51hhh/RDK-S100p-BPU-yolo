import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('volleyball_catch_controller'),
        'config',
        'nx_time_sync.yaml',
    )
    return LaunchDescription(
        [
            Node(
                package='volleyball_catch_controller',
                executable='nx_time_sync_publisher',
                name='nx_time_sync_publisher',
                output='screen',
                parameters=[config],
            )
        ]
    )
