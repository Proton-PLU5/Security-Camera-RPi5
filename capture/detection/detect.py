from dataclasses import dataclass
import threading
from time import time
from typing import Any, Optional, Tuple
from data.storage import StorageThread
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, mailbox, detection_store: DetectionStore, model, storage_thread: StorageThread):
        super().__init__(daemon=True, name="DetectionThread")
        self.mailbox = mailbox
        self.detection_store = detection_store
        self.model = model
        self.storage_thread = storage_thread
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            item = self.mailbox.get(timeout=0.1)
            if item is None:
                continue
            
            lowres_frame, timestamp, clip_id = item
            results = self.model(lowres_frame)
            self.detection_store.update(
                Detection(timestamp=timestamp, boxes=results, frame_size=lowres_frame.shape[:2])
            )

            for result in results:
                for box in result.boxes.xyxy:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox_x, bbox_y, bbox_width, bbox_height = map(int, box.xywh[0])

                    # Log the detection
                    logger.info(f"Detection: {class_name} (confidence: {confidence:.2f}) at "
                                f"({bbox_x}, {bbox_y}, {bbox_width}, {bbox_height}) in clip {clip_id}")

                    # Store the detection result in the storage thread
                    self.storage_thread.insert_detection(clip_id=clip_id, class_name=class_name, confidence=confidence,
                                bbox_x=bbox_x, bbox_y=bbox_y, bbox_width=bbox_width, bbox_height=bbox_height)

    def stop(self):
        self.stop_event.set()