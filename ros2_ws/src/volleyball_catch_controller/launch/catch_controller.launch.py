from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('volleyball_catch_controller'), 'config', 'catch_controller.yaml')
    return LaunchDescription([Node(package='volleyball_catch_controller', executable='catch_controller', name='catch_controller', output='screen', parameters=[cfg])])
