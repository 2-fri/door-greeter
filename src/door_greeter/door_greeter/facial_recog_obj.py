# Facenet Imports
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T
from PIL import Image as PImage
import cv2

# SQL Imports
import sqlite3, sqlite_vec
import numpy as np

# Our Imports
from door_greeter.llm_layer import Converser

# Global Settings
FACE_MARGIN = 20
SIMILARITY_THRESHOLD = 1.0
RECOGNITION_PATIENCE = 5
FORGETTING_PATIENCE = 10

# Facial Recognition Object
class FacialRecogObj():
    patience = RECOGNITION_PATIENCE
    person_memory = None #[embedding, rowid, forget]
    counter = 0 # For displaying multiple faces

    def __init__(self, yolo):
        # Reference to the yolo node
        self.yolo = yolo

        # Facenet Init
        self.mtcnn = MTCNN(image_size=160, margin=FACE_MARGIN)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        # SQL Init
        self.faces = sqlite3.connect("faces.db")
        sqlite_vec.load(self.faces)
        self.faces.execute("CREATE VIRTUAL TABLE IF NOT EXISTS faces USING vec0(embedding FLOAT[512], value TEXT);")

        # Internal Init
        self.person_memory = []
        self.llm_layer = Converser()

        print("facial_recog_object Initialized")
    
    def average_embeddings(self, em1, em2):
        avg = (em1 + em2) / 2.0
        return avg / np.linalg.norm(avg)

    def remember_person(self, embedding : np.ndarray, rowid : int, description : str = ""):
        for entry in self.person_memory:
            if entry[1] == rowid:
                entry[0] = self.average_embeddings(entry[0], embedding)
                entry[2] = FORGETTING_PATIENCE
                print(f"Person {rowid} RE-ENTERED the frame (embedding updated)")
                return
        self.person_memory.append([embedding, rowid, FORGETTING_PATIENCE])
        print(f"Person {rowid} ENTERED the frame")
        self.llm_layer.add_person(rowid, description)

    def advance_forgetting(self):
        self.counter = 0
        for num, entry in enumerate(self.person_memory[:]):
            entry[2] -= 1
            if entry[2] < 0:
                self.person_memory.pop(num)
                print(f"Person {entry[1]} LEFT the frame")
                description = self.llm_layer.remove_person(entry[1])
                if description:
                    self.faces.execute(
                        "UPDATE faces SET embedding = ?, value = ? WHERE rowid = ?", 
                        (entry[0].tobytes(), description, entry[1])
                    )
                    self.faces.commit()
                    print(f"Database ID {entry[1]} updated")

    def parse_face(self, person : np.ndarray):
        if person is None or person.ndim == 0:
            print("Null input to parse_face")
            return False
        cv2.imshow(f"person {self.counter}", person)
        try:
            face_crop = self.mtcnn(PImage.fromarray(person))
        except Exception as e:
            print(f'Exception during MTCNN processing caught: {e}')
            return False

        if face_crop is None:
            return False
        
        nparray_face = np.array(T.ToPILImage()(face_crop))
        self.counter += 1
        cv2.imshow(f'mtcnn face {self.counter}', nparray_face)
        if self.yolo.detect_people(nparray_face).shape[0] == 0: 
            print(f"Face {self.counter} invalid, discarding.")
            return False

        face_vect = self.resnet(face_crop.unsqueeze(0)).squeeze().detach().numpy()
        face_vect = face_vect / np.linalg.norm(face_vect)
        
        # COMPARE WITH MEMORY
        for entry in self.person_memory:
            if np.linalg.norm(face_vect - entry[0]) <= SIMILARITY_THRESHOLD: # Euclidean Dist
                entry[2] = FORGETTING_PATIENCE
                return True

        embedding = face_vect.tobytes()
        # Face Recognition
        find = self.faces.execute(
            "SELECT rowid, distance, embedding, value FROM faces WHERE embedding MATCH ? AND k = 1",
            (embedding,)
        ).fetchone()

        if find is None:
            if self.patience:
                self.patience -= 1
            else:
                self.faces.execute(
                    "INSERT INTO faces (embedding, value) VALUES (?, ?)",
                    (embedding, "")
                )
                self.faces.commit()
                self.remember_person(face_vect, 1)
        else:
            rowid, distance, old_embedding, description = find
            if distance <= SIMILARITY_THRESHOLD:    # Match
                self.patience = RECOGNITION_PATIENCE
                self.remember_person(self.average_embeddings(np.frombuffer(old_embedding, dtype=np.float32), face_vect), rowid, description)
            elif self.patience:                     # No Match, Waiting
                self.patience -= 1
            else:                                   # No Match, Adding
                self.faces.execute(
                    "INSERT INTO faces (embedding, value) VALUES (?, ?)",
                    (embedding, "")
                )
                self.faces.commit()
                self.patience = RECOGNITION_PATIENCE
                self.remember_person(face_vect, self.faces.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
        return True