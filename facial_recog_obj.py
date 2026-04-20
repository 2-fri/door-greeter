# Facenet Imports
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T
from PIL import Image as PImage
import cv2

# SQL Imports
import sqlite3, sqlite_vec
import numpy as np

# Global Settings
FACE_MARGIN = 10
SIMILARITY_THRESHOLD = 1.0
RECOGNITION_PATIENCE = 50
FORGETTING_PATIENCE = 10

# Facial Recognition Object
class FacialRecogObj():
    patience = RECOGNITION_PATIENCE
    person_memory = None #[embedding, forget]

    def __init__(self):
        # Facenet Init
        self.mtcnn = MTCNN(image_size=160, margin=FACE_MARGIN)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        # SQL Init
        self.faces = sqlite3.connect("faces.db")
        sqlite_vec.load(self.faces)
        self.faces.execute("CREATE VIRTUAL TABLE IF NOT EXISTS faces USING vec0(embedding FLOAT[512]);")

        # Internal Init
        self.person_memory = []

        print("facial_recog_object Initialized")
    
    def remember_person(self, embedding):
        self.person_memory.append([embedding, FORGETTING_PATIENCE])

    def advance_forgetting(self):
        for entry in self.person_memory:
            entry[1] -= 1
            if entry[1] < 0:
                self.person_memory.remove(entry)

    def parse_face(self, person : np.ndarray):
        face_crop = self.mtcnn(PImage.fromarray(person))
        if face_crop is not None:
            # TODO Check if this is a face

            # cv2.imshow('mtcnn face', cv2.cvtColor(np.array(T.ToPILImage()(face_crop)), cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
            embedding = self.resnet(face_crop.unsqueeze(0)).squeeze().detach().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tobytes()
            
            # COMPARE WITH MEMORY
            for entry in self.person_memory:
                if np.linalg.norm(embedding - entry[0]) <= SIMILARITY_THRESHOLD: # Euclidean Dist
                    entry[1] = FORGETTING_PATIENCE
                    return

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
                    self.remember_person(embedding)
            else:
                rowid, distance = find
                if distance <= SIMILARITY_THRESHOLD:    # Match
                    print(f"Recognized {rowid} with dist {distance}")
                    self.patience = RECOGNITION_PATIENCE
                    self.remember_person(embedding)
                elif self.patience:                     # No Match, Waiting
                    self.patience -= 1
                    print(f"New Face with dist {distance} -> Waiting ({self.patience}/{RECOGNITION_PATIENCE})")
                else:                                   # No Match, Adding
                    print(f"New Face with dist {distance} -> Adding")
                    self.faces.execute(
                        "INSERT INTO faces (embedding) VALUES (?)",
                        (embedding,)
                    )
                    self.faces.commit()
                    self.patience = RECOGNITION_PATIENCE
                    self.remember_person(embedding)