from setuptools import setup

package_name = 'volleyball_catch_controller'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/catch_controller.yaml',
            'config/catch_controller_d435_only.yaml',
            'config/d435_throw_analysis.yaml',
            'config/d435_throw_ground_truth.csv',
            'config/goal_to_cmd_vel.yaml',
            'config/nx_time_sync.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/catch_controller.launch.py',
            'launch/d435_only_chase.launch.py',
            'launch/nx_time_sync.launch.py',
        ]),
    ],
    install_requires=['setuptools', 'numpy'], zip_safe=True, maintainer='sunrise',
    maintainer_email='sunrise@example.com', description='Volleyball catch controller',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'catch_controller = volleyball_catch_controller.node:main',
        'goal_to_cmd_vel = volleyball_catch_controller.goal_to_cmd_vel_node:main',
        'nx_time_sync_publisher = volleyball_catch_controller.nx_time_sync_node:main',
        'prepare_nx_epoch = volleyball_catch_controller.nx_epoch:main',
        'check_chassis_odom = volleyball_catch_controller.chassis_odom_check_node:main',
        'transport_recovery_test = volleyball_catch_controller.transport_recovery_test_node:main',
        'analyze_d435_throw_bag = volleyball_catch_controller.analyze_d435_throw_bag:main',
    ]},
)
