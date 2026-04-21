# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# YOLO Imports
from ultralytics import YOLO
import numpy as np

# Our Imports
from facial_recog_obj import FacialRecogObj

# Global Setting
YOLO_MODEL = "yolo11s.pt"

# YOLO Node
class YoloNode(Node):
    def __init__(self):
        # ROS Init
        super().__init__('yolo_node')
        self.image_sub = self.create_subscription(
            Image,
            'camera_raw',
            self.listener_callback,
            1
        )
        self.bridge = CvBridge()

        # YOLO Init
        self.model = YOLO(YOLO_MODEL)

        # Create Facial Recog Objectect
        self.facial_recog_obj = FacialRecogObj()

        print("YOLO Node Initialized")
    
    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        cv2.imshow('camera', frame)
        result = self.model.predict(frame, classes = [0], save = False, verbose = False)[0]

        # Person Segmentation
        halfway_width = int(frame.shape[1] / 2)
        # print(halfway_width)
        person_central_x = 0 # [x,y] avg of all people in frame
        person_count = 0
        rotation_angle = 0 # set to no rotation at first, depending if we see people will change

        for num, box in enumerate(result.boxes):
            coords = [int(i) for i in box.xyxy[0].tolist()] # Get bounding box, convert all values to ints
            # print(coords)
            
            person = frame[coords[1]:coords[3],coords[0]:coords[2]]

            if self.facial_recog_obj.parse_face(person):
                person_central_x += box.xywh[0].tolist()[0]
                person_count += 1
            else:
                print("cant find person")
        
        if person_count > 0:
            person_central_x = int(person_central_x / person_count)
        
        print(person_central_x)
        if person_central_x - halfway_width > 0:
            print("turn right")
        else:
            print("turn left")
        self.facial_recog_obj.advance_forgetting()
        cv2.waitKey(1)

        # Turn to Person

        

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = YoloNode()
    rclpy.spin(yolo_subscriber)

    yolo_subscriber.destroy_node()
    rclpy.shutdown()

main()