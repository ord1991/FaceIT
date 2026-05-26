import os
import cv2
import numpy as np
from deepface import DeepFace
from sqlalchemy.orm import Session
from database import User, SessionLocal
import time

class FaceEngine:
    def __init__(self, model_name="VGG-Face", detector_backend="opencv", threshold=0.4):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.threshold = threshold
        self.known_users = []
        self.load_known_users()

    def load_known_users(self):
        db = SessionLocal()
        try:
            self.known_users = db.query(User).all()
            print(f"Loaded {len(self.known_users)} users from database.")
        finally:
            db.close()

    def cosine_distance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.matmul(np.transpose(source_representation), source_representation)
        c = np.matmul(np.transpose(test_representation), test_representation)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def process_frame(self, frame):
        # Result list to store detections
        results = []

        try:
            # Face detection and representation
            # enforce_detection=False to avoid exceptions when no face is found
            faces = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            print(f"Error in DeepFace.represent: {e}")
            return frame, []

        for face in faces:
            # DeepFace returns facial_area even if no face is detected if enforce_detection is False?
            # Actually, if no face is detected, it might return a representation of the whole image.
            # We should check if the confidence or area is reasonable if possible,
            # but usually DeepFace.represent returns a list of face objects.

            # If enforce_detection=False, and no face is detected, it returns a single item list
            # with the whole image as the face area.
            # However, recent DeepFace versions might behave differently.

            # Let's check for face detection by looking at the facial_area.
            # If opencv backend is used, it returns x, y, w, h.
            area = face["facial_area"]

            # If the "face" is the whole image, it's likely no face was detected.
            if area["w"] == frame.shape[1] and area["h"] == frame.shape[0]:
                continue

            embedding = face["embedding"]
            best_match = None
            min_dist = 1.0

            for user in self.known_users:
                dist = self.cosine_distance(user.embedding, embedding)
                if dist < min_dist:
                    min_dist = dist
                    best_match = user

            status = "Unknown"
            color = (0, 255, 255) # Yellow
            name = "Unknown"

            if best_match and min_dist < self.threshold:
                name = best_match.name
                if best_match.status == "approved":
                    status = "Approved"
                    color = (0, 255, 0) # Green
                elif best_match.status == "blacklisted":
                    status = "Blacklisted"
                    color = (0, 0, 255) # Red

            # Draw bounding box
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({status})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results.append({
                "name": name,
                "status": status,
                "distance": float(min_dist),
                "box": [x, y, w, h],
                "embedding": embedding
            })

        return frame, results

    def get_embedding(self, img_path):
        try:
            results = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            if results:
                return results[0]["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
        return None
