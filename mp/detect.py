import ctypes
import json
from multiprocessing import Array, Process, Event, Queue, Value
from mp.capture import CaptureBuffer
from mp.storage import Task, TaskFactory
from ultralytics import YOLO # type: ignore
from data.metrics import metrics
import ncnn
import time

MAX_BYTES = 65536

class DetectionBuffer:
    def __init__(self, max_bytes=MAX_BYTES):
        self.max_bytes = max_bytes
        self.buf_a = Array(ctypes.c_char, max_bytes, lock=False)
        self.buf_b = Array(ctypes.c_char, max_bytes, lock=False)
        self.len_a = Value('i', 0, lock=False)
        self.len_b = Value('i', 0, lock=False)

        self.active = Value('i', 0)  # 0 = buf_a is readable, 1 = buf_b is readable
        self.version = Value('l', 0)

    def get(self) -> list:
        # Return the latest detections snapshot
        active = self.active.value
        buf = self.buf_a if active == 0 else self.buf_b
        length_holder = self.len_a if active == 0 else self.len_b
        n = length_holder.value

        if n == 0: # No detections available
            return []
        
        payload = bytes(buf[:n]) # type: ignore
        return json.loads(payload.decode("utf-8"))

    def get_with_version(self) -> tuple[list, int]:
        # Return the latest detections snapshot along with its version
        detections = self.get()
        version = self.version.value
        return detections, version

    def write(self, detections: list):
        payload = json.dumps(detections).encode("utf-8")
        if len(payload) > self.max_bytes:
            raise ValueError(f"Detections payload ({len(payload)} bytes) exceeds max_bytes ({self.max_bytes})")

        active = self.active.value
        target_buf = self.buf_b if active == 0 else self.buf_a
        target_len = self.len_b if active == 0 else self.len_a

        target_buf[:len(payload)] = payload # type: ignore
        target_len.value = len(payload)  # commit length only after bytes are written

        with self.active.get_lock():
            self.active.value = 1 - self.active.value
        with self.version.get_lock():
            self.version.value += 1

class DetectProcess(Process):
    def __init__(self, 
                 stop_event : Event, # type: ignore
                 capture_buffer : CaptureBuffer,
                 storage_task_queue : "Queue[Task]",
                 detection_buffer : DetectionBuffer,
                 ):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.capture_buffer = capture_buffer
        self.storage_task_queue = storage_task_queue
        self.detection_buffer = detection_buffer

    def configure_ncnn(self):
        # NCNN THREAD LIMITING
        _orig_load_param = ncnn.Net.load_param # type: ignore

        def _load_param_with_thread_limit(self, path):
            self.opt.num_threads = 1  # set before weights get packed
            return _orig_load_param(self, path)

        ncnn.Net.load_param = _load_param_with_thread_limit # type: ignore

    def run(self):
        self.configure_ncnn()
        self.model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model  
        self.task_factory = TaskFactory()
        
        while self.stop_event.is_set() == False:
            frame = self.capture_buffer.get()
            if frame is None:
                continue

            with metrics.time("detection_inference_time"):
                results = self.model(frame)  # Perform detection on the frame

            # Process results and create tasks for storage
            detections = []
            with metrics.time("detection_processing_time"):
                for result in results:
                    for box in result.boxes:
                        clip_id = "some_clip_id"  # You would get this from your capture logic
                        timestamp = time.time()  # Current timestamp
                        class_name = result.names[int(box.cls.item())]
                        confidence = float(box.conf.item())
                        bbox_x, bbox_y, bbox_width, bbox_height = box.xywh[0].tolist()

                        task = self.task_factory.insert_detection(
                            clip_id=clip_id,
                            timestamp=timestamp,
                            class_name=class_name,
                            confidence=confidence,
                            bbox_x=bbox_x,
                            bbox_y=bbox_y,
                            bbox_width=bbox_width,
                            bbox_height=bbox_height
                        )

                        self.storage_task_queue.put(task)  # Send task to storage process
                        
                        detections.append({
                            "clip_id": clip_id,
                            "timestamp": timestamp,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox_x": bbox_x,
                            "bbox_y": bbox_y,
                            "bbox_width": bbox_width,
                            "bbox_height": bbox_height
                        })

                self.detection_buffer.write(detections)  # Write detections to shared buffer
