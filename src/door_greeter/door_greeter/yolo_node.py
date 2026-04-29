# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

# YOLO Imports
from ultralytics import YOLO
import numpy as np

# Our Imports
from door_greeter.facial_recog_obj import FacialRecogObj
from door_greeter.movement_obj import MovementObj

# Global Setting
YOLO_MODEL = "yolo11s.pt"

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
        self.bridge = CvBridge()

        # YOLO Init
        self.model = YOLO(YOLO_MODEL)

        # Create Subobjects
        self.facial_recog_obj = FacialRecogObj(self)
        self.movement_obj = MovementObj(self)

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
        return np.array([X, Y, Z])

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
        avg_pos = np.zeros(3)
        person_count = 0

        for box in self.detect_people(frame):
            coords = [int(i) for i in box.xyxy[0].tolist()]
            person = frame[coords[1]:coords[3],coords[0]:coords[2]]
            self.facial_recog_obj.parse_face(person)

            center = [int(i) for i in box.xywh[0].tolist()]
            person_pos = self.get_3d_position(center[0], center[1])
            if person_pos is not None:
                avg_pos += person_pos
                person_count += 1

            if person_count > 0:
                self.movement_obj.update_with_person_position(avg_pos / person_count)
            else:
                self.movement_obj.update_with_empty_frame()

        self.facial_recog_obj.advance_forgetting()
        cv2.waitKey(1)     

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = YoloNode()
    rclpy.spin(yolo_subscriber)

    yolo_subscriber.destroy_node()
    rclpy.shutdown()

main()
