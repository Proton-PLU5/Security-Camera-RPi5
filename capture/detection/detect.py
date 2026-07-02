from dataclasses import dataclass
import threading
from time import time
from typing import Any, Optional, Tuple

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
    def __init__(self, mailbox, detection_store: DetectionStore, model):
        super().__init__(daemon=True, name="DetectionThread")
        self.mailbox = mailbox
        self.detection_store = detection_store
        self.model = model
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            item = self.mailbox.get(timeout=0.1)
            if item is None:
                continue
            
            timestamp, frame = item
            results = self.model(frame)
            self.detection_store.update(
                Detection(timestamp=timestamp, boxes=results, frame_size=frame.shape[:2])
            )

    def stop(self):
        self.stop_event.set()