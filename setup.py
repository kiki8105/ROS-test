from setuptools import setup

setup(
    name='my_turtle_pkg',
    version='0.0.1',
    packages=['my_turtle_pkg'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/my_turtle_pkg']),
        ('share/my_turtle_pkg', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kiki',
    maintainer_email='kiki@example.com',
    description='TurtleBot3 circular motion practice node',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'circle_turtle = my_turtle_pkg.circle_turtle:main',
        'scan_drive = my_turtle_pkg.scan_drive:main',
        'lidar_mock = my_turtle_pkg.lidar_mock:main',
        'action_publisher = my_turtle_pkg.action_publisher:main',
    ]},
)
