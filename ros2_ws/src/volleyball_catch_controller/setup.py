from setuptools import setup

package_name = 'volleyball_catch_controller'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/catch_controller.yaml']),
        ('share/' + package_name + '/launch', ['launch/catch_controller.launch.py']),
    ],
    install_requires=['setuptools', 'numpy'], zip_safe=True, maintainer='sunrise',
    maintainer_email='sunrise@example.com', description='Volleyball catch controller',
    license='Apache-2.0',
    entry_points={'console_scripts': ['catch_controller = volleyball_catch_controller.node:main']},
)
