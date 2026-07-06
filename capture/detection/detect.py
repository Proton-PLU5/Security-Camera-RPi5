from dataclasses import dataclass
import threading
from time import time
from typing import Any, Optional, Tuple
from data.metrics import metrics
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
            results = self.model(lowres_frame, verbose=False)
            for stage,ms in results[0].speed.items():
                metrics.record(f"detection_{stage}", ms / 1000.0)  # convert to seconds

            detected_at = time()
                
            self.detection_store.update(
                Detection(timestamp=timestamp, boxes=results, frame_size=lowres_frame.shape[:2])
            )

            if clip_id is None:
                continue  # No active clip, skip storage

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
                    
    def stop(self):
        self.stop_event.set()