# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

# YOLO Imports
from ultralytics import YOLO
import numpy as np

# Our Imports
from door_greeter.facial_recog_obj import FacialRecogObj

# Global Setting
YOLO_MODEL = "yolo11s.pt"
VELOCITY_CONSTANT = 0.5

# YOLO Node
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')


        # ROS Paremeters
        self.declare_parameter('movement_output', False)
        self.declare_parameter('camera_topic', 'k4a/rgb/image_raw')
        self.movement_output = self.get_parameter('movement_output').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # ROS Init
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.listener_callback,
            1
        )
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.twist = Twist()
        self.bridge = CvBridge()

        # YOLO Init
        self.model = YOLO(YOLO_MODEL)

        # Create Facial Recog Objectect
        self.facial_recog_obj = FacialRecogObj(self)

        print(f"yolo_node Initialized\n\tmovement_output = {self.movement_output}")
    
    def detect_people(self, frame):
        return self.model.predict(frame, classes = [0], save = False, verbose = False)[0].boxes

    def listener_callback(self, msg):
        received = self.bridge.imgmsg_to_cv2(msg)
        frame = received[:, :, :3]
        cv2.imshow('camera', frame)

        # Person Segmentation
        halfway_width = int(frame.shape[1] / 2)
        person_central_x = 0 # [x,y] avg of all people in frame
        person_count = 0
        rotation_vel = 0 # set to no rotation at first, depending if we see people will change

        for num, box in enumerate(self.detect_people(frame)):
            coords = [int(i) for i in box.xyxy[0].tolist()] # Get bounding box, convert all values to ints
            
            person = frame[coords[1]:coords[3],coords[0]:coords[2]]

            # if self.facial_recog_obj.parse_face(person):
            person_central_x += box.xywh[0].tolist()[0]
            person_count += 1
        
        if person_count > 0:
            person_central_x = int(person_central_x / person_count)   
            rotation_vel = ((person_central_x - halfway_width) / halfway_width) * VELOCITY_CONSTANT * -1
            if self.movement_output: # movement_output = True
                print(person_central_x)
                print(halfway_width)
                print(rotation_vel)
            if rotation_vel < 0.2 and rotation_vel > -0.2:
                rotation_vel = 0.0
            
            self.twist.angular.z = rotation_vel

            if self.movement_output: # movement_output = True
                if rotation_vel > 0:
                    print("turn right")
                elif rotation_vel < 0:
                    print("turn left")
                else:
                    print("stationary")
        else:
            self.twist.angular.z = 0.0
        self.facial_recog_obj.advance_forgetting()
        cv2.waitKey(1)

        # Turn to Person
        self.vel_publisher.publish(self.twist)        

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = YoloNode()
    rclpy.spin(yolo_subscriber)

    yolo_subscriber.destroy_node()
    rclpy.shutdown()

main()
