## Debug Mode:

To run this code, run ```./script.sh```

Then, run these commands on different terminals:

``` bash
python camera_publisher_node.py
python yolo_node.py
```

To authenticate output, run:

``` bash
python output_authenticator.py
```

## ROS Build

To build this package...

To run segway motor:
ros2 launch segway_rmp_ros2 segway_rmp_ros2.launch.py 

To run azure kinect cam:
ros2 run azure_kinect_ros2_driver azure_kinect_node

To run lib core:
ros2 run door_greeter core
