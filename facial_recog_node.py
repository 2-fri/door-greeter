# Facenet Imports
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T
from PIL import Image as PImage

# ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# SQL Imports
import sqlite3, sqlite_vec
import numpy as np

# Global Constants
SIMILARITY_THRESHOLD = 1
RECOGNITION_PATIENCE = 50

# Facial Recognition Node
class FacialRecogNode(Node):
    patience = RECOGNITION_PATIENCE

    def __init__(self):
        # ROS Init
        super().__init__('facial_recog_node')
        self.image_sub = self.create_subscription(
            Image,
            'camera_raw',
            self.listener_callback,
            1
        )
        self.bridge = CvBridge()

        # Facenet Init
        self.mtcnn = MTCNN(image_size=160, margin=10)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        # SQL Init
        self.faces = sqlite3.connect("faces.db")
        sqlite_vec.load(self.faces)
        self.faces.execute("CREATE VIRTUAL TABLE IF NOT EXISTS faces USING vec0(embedding FLOAT[512]);")

        print("facial_recog_node Initialized")

    
    def listener_callback(self, msg):
        face_crop = self.mtcnn(PImage.fromarray(self.bridge.imgmsg_to_cv2(msg)))
        if face_crop is not None:
            #Check if this is a face

            cv2.imshow('horrible', cv2.cvtColor(np.array(T.ToPILImage()(face_crop)), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            embedding = self.resnet(face_crop.unsqueeze(0)).squeeze().detach().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tobytes()
            
            # Face Recognition
            find = self.faces.execute(
                "SELECT rowid, distance FROM faces WHERE embedding MATCH ? AND k = 1",
                (embedding,)
            ).fetchone()

            if find is None:
                if self.patience:
                    self.patience -= 1
                    print(f"Empty Database -> Waiting ({self.patience}/{RECOGNITION_PATIENCE})")
                else:
                    print(f"Empty Database -> Adding")
                    self.faces.execute(
                        "INSERT INTO faces (embedding) VALUES (?)",
                        (embedding,)
                    )
                    self.faces.commit()
            else:
                rowid, distance = find
                if distance <= SIMILARITY_THRESHOLD:
                    # Found :)
                    print(f"I recognize you {rowid} with dist {distance}")
                    self.patience = RECOGNITION_PATIENCE
                else:
                    # New
                    if self.patience:
                        self.patience -= 1
                        print(f"New Face with dist {distance} -> Waiting ({self.patience}/{RECOGNITION_PATIENCE})")
                    else:
                        print(f"New Face with dist {distance} -> Adding")
                        self.faces.execute(
                            "INSERT INTO faces (embedding) VALUES (?)",
                            (embedding,)
                        )
                        self.faces.commit()
                        self.patience = RECOGNITION_PATIENCE
                

            


def main(args=None):
    rclpy.init(args=args)
    face_recog_subscriber = FacialRecogNode()
    rclpy.spin(face_recog_subscriber)

    face_recog_subscriber.destroy_node()
    rclpy.shutdown()

main()