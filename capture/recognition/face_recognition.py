from dataclasses import dataclass
import threading
from typing import Optional

import cv2
import numpy as np
import face_recognition
from capture.mailbox import MailBox
from capture.recognition.face_queue import SnapshotAtomicQueue
from data.storage import StorageThread
from firmware.config import Config
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self, config: Config):
        self.face_encodings_dir = config.getString("known_face_encodings_directory", "./face/encodings")
        self.tolerance = config.getFloat("face_recognition_tolerance", 0.6)
        self.known_face_encodings, self.known_face_names = self.load_known_face_encodings_and_names(self.face_encodings_dir)
        self.config = config

    def load_known_face_encodings_and_names(self, encodings_dir):
        known_face_encodings = []
        known_face_names = []
        if not os.path.exists(encodings_dir):
            os.makedirs(encodings_dir)
            return known_face_encodings, known_face_names

        for filename in os.listdir(encodings_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(encodings_dir, filename), "rb") as f:
                    encoding = pickle.load(f)
                    known_face_encodings.append(encoding)
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)

        return known_face_encodings, known_face_names

    def add_known_face(self, image : np.ndarray, name: str):
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            logger.warning(f"No faces found in the image.")
            return
        
        if len(face_encodings) > 1:
            logger.warning(
                "%d faces detected - using the first one for '%s'",
                len(face_encodings), name,
            )

        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)

        # Save the encoding to a pickle file
        with open(os.path.join(self.face_encodings_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(face_encodings[0], f)

    def recognise(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        recognised_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if self.known_face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_index = int(np.argmin(distances))
                
                if distances[best_index] <= self.tolerance:
                    name = self.known_face_names[best_index]

            recognised_faces.append({
                "name": name,
                "location": (top, right, bottom, left)
            })

        return recognised_faces

@dataclass
class FaceCropJob:
    crop: np.ndarray
    crop_origin: tuple[int, int]
    timestamp: float
    clip_id: Optional[str]

class FaceRecognitionThread(threading.Thread):

    def __init__(self, queue: SnapshotAtomicQueue, 
                 face_recognition: FaceRecognition,
                 storage_thread: StorageThread):
        super().__init__(daemon=True, name="FaceRecognitionThread")
        self.queue = queue
        self.face_recognition = face_recognition
        self.storage_thread = storage_thread
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            
            job : Optional[FaceCropJob] = self.queue.get()  # This will block until a new job is available

            if job is None:
                continue  # No job available, continue the loop

            try:
                results = self.face_recognition.recognise(job.crop)
            except Exception:
                logger.error(f"Error during face recognition.")
                continue
            
            if job.clip_id is None:
                continue  # No active clip, skip storage

            self.storage_thread.insert_recognition(
                clip_id=job.clip_id,
                timestamp=job.timestamp,
                name=results[0]["name"] if results else "Unknown"
            )

    def stop(self):
        self.stop_event.set()