import cv2
import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge

# Global Settings
PUBLISHING_PERIOD = 2.0

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.camDeviceNum = 0
        self.camera = cv2.VideoCapture(self.camDeviceNum)

        self.bridge_obj = CvBridge()
        self.publisher = self.create_publisher(Image, 'camera_raw', 1)
        self.timer = self.create_timer(PUBLISHING_PERIOD, self.callback_func)

        print("Camera Publisher Initialized")

    def callback_func(self):
        success, frame = self.camera.read()
        
        if success:
            self.publisher.publish(self.bridge_obj.cv2_to_imgmsg(frame))


def main(args=None):
    rclpy.init(args=args)
    camera_pub = CameraPublisher()
    rclpy.spin(camera_pub)

    CameraPublisher.destroy_node()
    rclpy.shutdown()

main()