from dataclasses import dataclass
import threading
from time import time
from typing import Any, Optional, Tuple

import numpy as np
from capture.mailbox import MailBox
from capture.recognition.face_queue import SnapshotAtomicQueue
from capture.recognition.face_recognition import FaceCropJob
from data.metrics import metrics
from data.storage import StorageThread
import logging

logger = logging.getLogger(__name__)


def crop_person_for_face_recognition(
    frame : np.ndarray,
    lowres_box_xyxy : Tuple[int, int, int, int],
    lowres_size: Tuple[int, int],
    pad_ratio: float = 0.2,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop a person from the high-resolution frame for face recognition.
    """

    main_height, main_width = frame.shape[:2]   
    lowres_width, lowres_height = lowres_size
    scale_x = main_width / lowres_width
    scale_y = main_height / lowres_height

    x1, y1, x2, y2 = lowres_box_xyxy
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y

    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * pad_ratio, box_h * pad_ratio

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(main_width, int(x2 + pad_x))
    crop_y2 = min(main_height, int(y2 + pad_y))
 
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, (crop_x1, crop_y1)

@dataclass
class Detection:
    timestamp: int  # SensorTimestamp (ns) of the source frame
    boxes: Any       # whatever your YOLO wrapper returns
    frame_size: Tuple[int, int]

class DetectionStore:
    """Thread-safe holder for the most recent YOLO result."""
 
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[Detection] = None
 
    def update(self, detection: Detection):
        with self._lock:
            self._latest = detection
 
    def latest(self) -> Optional[Detection]:
        with self._lock:
            return self._latest
        
class DetectionThread(threading.Thread):
    def __init__(self, mailbox : MailBox, 
                 detection_store: DetectionStore, 
                 model, 
                 storage_thread: StorageThread,
                 face_queue : Optional[SnapshotAtomicQueue] = None):
        super().__init__(daemon=True, name="DetectionThread")
        self.mailbox = mailbox
        self.detection_store = detection_store
        self.model = model
        self.storage_thread = storage_thread
        self.face_queue = face_queue
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            item = self.mailbox.get(timeout=0.1)
            if item is None:
                continue
            
            lowres_frame, stream_frame, timestamp, clip_id = item
            results = self.model(lowres_frame, verbose=False)
            for stage,ms in results[0].speed.items():
                metrics.record(f"detection_{stage}", ms / 1000.0)  # convert to seconds

            detected_at = time() * 1000
                
            self.detection_store.update(
                Detection(timestamp=timestamp, boxes=results, frame_size=lowres_frame.shape[:2])
            )

            if clip_id is None:
                continue  # No active clip, skip storage
            
            any_person_detected = False

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf.item())

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_x, bbox_y = int(x1), int(y1)
                    bbox_width, bbox_height = int(x2 - x1), int(y2 - y1)

                    self.storage_thread.insert_detection(
                        clip_id=clip_id,
                        timestamp=detected_at,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_x=bbox_x,
                        bbox_y=bbox_y,
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                    )

                    # Only crop+queue for face recognition on person detections
                    # And only if a face mailbox is provided and is empty
                    if (
                        self.face_queue is not None and
                        class_name == "person"
                    ):
                        any_person_detected = True

                        crop, crop_origin = crop_person_for_face_recognition(
                            frame=stream_frame,
                            lowres_box_xyxy=(bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height),
                            lowres_size=lowres_frame.shape[1::-1],
                        )

                        if crop.size > 0:
                            self.face_queue.add(FaceCropJob(
                                crop=crop,
                                crop_origin=crop_origin,
                                timestamp=detected_at,
                                clip_id=clip_id
                            ))
            
            if self.face_queue is not None and any_person_detected == True:
                self.face_queue.lock()

    def stop(self):
        self.stop_event.set()