from setuptools import find_packages, setup

package_name = 'door_greeter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'ultralytics', 'pyaudio'],
    zip_safe=True,
    maintainer='2_fri',
    maintainer_email='as228976@eid.utexas.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher = door_greeter.camera_publisher_node:main',
            'core = door_greeter.yolo_node:main',
            'debug = door_greeter.output_authenticator:main',
            'check_db = door_greeter.database_checker:check_db',
        ],
    },
)
