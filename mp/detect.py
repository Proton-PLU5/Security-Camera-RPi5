from multiprocessing import Process, Event, Queue
from mp.capture import CaptureBuffer
from mp.detection_buffer import DetectionBuffer
from mp.storage import Task, TaskFactory
from ultralytics import YOLO # type: ignore
from data.metrics import metrics
import ncnn
import time
import logging

class DetectProcess(Process):
    def __init__(self, 
                 stop_event : Event, # type: ignore
                 capture_buffer : CaptureBuffer,
                 storage_task_queue : "Queue[Task]",
                 detection_buffer : DetectionBuffer,
                 lowres_size: tuple[int, int] = (960, 544)
                 ):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.capture_buffer = capture_buffer
        self.storage_task_queue = storage_task_queue
        self.detection_buffer = detection_buffer
        self.lowres_size = lowres_size

    def configure_ncnn(self):
        # NCNN THREAD LIMITING
        _orig_load_param = ncnn.Net.load_param # type: ignore

        def _load_param_with_thread_limit(self, path):
            self.opt.num_threads = 2  # set before weights get packed
            return _orig_load_param(self, path)

        ncnn.Net.load_param = _load_param_with_thread_limit # type: ignore

    def run(self):
        self.configure_ncnn()
        self.model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model  
        self.task_factory = TaskFactory()
        
        logger = logging.getLogger(__name__)
        logger.info("Detection process started.")
        
        try:
            while self.stop_event.is_set() == False:
                frame, clip_id = self.capture_buffer.get()
                if frame is None:
                    continue


                with metrics.time("detection_inference_time"):
                    results = self.model(frame, verbose=False)  # Perform detection on the frame

                # Process results and create tasks for storage
                detections = []
                with metrics.time("detection_processing_time"):
                    for result in results:
                        for box in result.boxes:
                            timestamp = time.time()  # Current timestamp
                            class_name = result.names[int(box.cls.item())]
                            confidence = float(box.conf.item())

                            if confidence < 0.5:  # Filter out low-confidence detections
                                continue

                            bbox_x, bbox_y, bbox_width, bbox_height = box.xywh[0].tolist()
                            
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
        except Exception as e:
            logger.info(f"Error in detection process: {e}")
        
