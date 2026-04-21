# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# YOLO Node
class AuthenticatorNode(Node):
    def __init__(self):
        # ROS Init
        super().__init__('authenticator_node')
        self.image_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            1
        )
    
    def listener_callback(self, msg):
        self.get_logger().info(f'Velocity broadcasted = {msg.angular}')


def main(args=None):
    rclpy.init(args=args)
    auth_subscriber = AuthenticatorNode()
    rclpy.spin(auth_subscriber)

    auth_subscriber.destroy_node()
    rclpy.shutdown()

main()