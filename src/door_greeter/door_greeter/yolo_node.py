# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
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
VELOCITY_CONSTANT = 0.3

# YOLO Node
class YoloNode(Node):
    depth_image = None
    fx = fy = cx = cy = None

    def __init__(self):
        super().__init__('yolo_node')

        # ROS Paremeters
        self.declare_parameter('movement_output', False)
        self.declare_parameter('camera_topic', 'k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', 'k4a/depth_to_rgb/image_raw')
        self.declare_parameter('info_topic', 'k4a/rgb/camera_info')
        self.movement_output = self.get_parameter('movement_output').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.info_topic = self.get_parameter('info_topic').get_parameter_value().string_value

        # ROS Init
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 1)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 1)
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.twist = Twist()
        self.bridge = CvBridge()
        self.rotation_amt = 0.0

        # YOLO Init
        self.model = YOLO(YOLO_MODEL)

        # Create Facial Recog Objectect
        self.facial_recog_obj = FacialRecogObj(self)

        print(f"yolo_node Initialized\n\tmovement_output = {self.movement_output}\n\tcamera_topic = {self.camera_topic}\n\tdepth_topic = {self.depth_topic}")

    def detect_people(self, frame):
        return self.model.predict(frame, classes = [0], save = False, verbose = False)[0].boxes

    def get_3d_position(self, u : int, v : int):
        if self.depth_image is None or self.fx is None:
            self.get_logger().info("Depth information not available yet!")
            return
        depth = self.depth_image[v, u]
        Z = float(depth) / 1000.0
        if Z == 0:
            return
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y, Z

    def info_callback(self, msg : CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg : Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg : Image):
        received = self.bridge.imgmsg_to_cv2(msg)
        frame = received[:, :, :3]
        cv2.imshow('camera', frame)

        # Person Segmentation
        halfway_width = int(frame.shape[1] / 2)
        person_central_x = 0 # [x,y] avg of all people in frame
        person_count = 0
        rotation_vel = 0 # set to no rotation at first, depending if we see people will change

        for box in self.detect_people(frame):
            coords = [int(i) for i in box.xyxy[0].tolist()] # Get bounding box, convert all values to ints
            
            person = frame[coords[1]:coords[3],coords[0]:coords[2]]

            self.facial_recog_obj.parse_face(person)
            center = box.xywh[0].tolist()
            person_central_x += center[0]
            person_count += 1
            # print(self.get_3d_position(int(center[0]), int(center[1])))
        
        if person_count > 0:
            person_central_x = int(person_central_x / person_count)   
            rotation_vel = ((person_central_x - halfway_width) / halfway_width) * VELOCITY_CONSTANT * -1
            if self.movement_output: # movement_output = True
                print(person_central_x)
                print(halfway_width)
                print(rotation_vel)
            if rotation_vel < 0.1 and rotation_vel > -0.1:
                rotation_vel = 0.0
            
            if self.movement_output: # movement_output = True
                if rotation_vel > 0:
                    print("turn right")
                elif rotation_vel < 0:
                    print("turn left")
                else:
                    print("stationary")
        else:
            if self.rotation_amt > 0.05:
                if self.rotation_amt > 0.1:
                    rotation_vel = -0.1
                else:
                    rotation_vel = -0.01
            elif self.rotation_amt < -0.05:
                if self.rotation_amt > -0.1:
                    rotation_vel = 0.1
                else:
                    rotation_vel = 0.01
            else:  
                rotation_vel = 0.0
        self.facial_recog_obj.advance_forgetting()
        cv2.waitKey(1)

        if abs(self.rotation_amt) > 3.0:
            rotation_vel = 0.0
        # Turn to Person
        self.twist.angular.z = rotation_vel
        self.rotation_amt += rotation_vel
        print(rotation_vel)
        print(self.rotation_amt)
        print("**")
        self.vel_publisher.publish(self.twist)        

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = YoloNode()
    rclpy.spin(yolo_subscriber)

    yolo_subscriber.destroy_node()
    rclpy.shutdown()

main()
