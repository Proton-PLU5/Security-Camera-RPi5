import asyncio
import logging
from multiprocessing import Event, Process, Queue, shared_memory, Value
import os
import time
import uuid
import numpy as np
from data.metrics import metrics
from mp.detect import DetectionBuffer
from mp.storage import Task, TaskFactory
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

logger = logging.getLogger(__name__)

class CaptureBuffer:
    def __init__(self, shape, dtype=np.uint8):
        self.shape = shape
        self.dtype = dtype
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        # Double buffering to avoid overwriting frames while processing
        self.shared_buffer_a = shared_memory.SharedMemory(create=True, size=nbytes)
        self.shared_buffer_b = shared_memory.SharedMemory(create=True, size=nbytes)

        self.active = Value('i', 0)  # 0 for buffer A, 1 for buffer B

        # Version counter to track updates
        # Important for consumers to know if they are reading an updated frame, to prevent
        # processes like YOLO and face recognition from processing the same frame multiple times (VERY BAD).
        self.version = Value('l', 0) 
        self.current_clip_id = uuid.uuid4().hex  # Track the current clip ID

    def get(self) -> tuple[np.ndarray, str]:
        # Return the active buffer as a numpy array.
        active = self.active.value
        buf = self.shared_buffer_a.buf if active == 0 else self.shared_buffer_b.buf
        return np.ndarray(self.shape, dtype=self.dtype, buffer=buf).copy(), self.current_clip_id

    def write(self, frame: np.ndarray, clip_id: str):
        # write to the inactive buffer, then flip
        active = self.active.value
        target_buf = self.shared_buffer_b.buf if active == 0 else self.shared_buffer_a.buf
        target = np.ndarray(self.shape, dtype=self.dtype, buffer=target_buf)
        target[:] = frame

        with self.active.get_lock():
            self.active.value = 1 - self.active.value
        with self.version.get_lock():
            self.version.value += 1
    
    def close(self):
        self.shared_buffer_a.close()
        self.shared_buffer_a.unlink()
        self.shared_buffer_b.close()
        self.shared_buffer_b.unlink()


class CaptureProcess(Process):
    def __init__(self, 
                 storage_task_queue : "Queue[Task]", 
                 stop_event: Event,  # type: ignore
                 capture_buffer: CaptureBuffer,
                 detection_buffer: DetectionBuffer,
                 db_path: str = 'storage.db',
                 clip_dir: str = 'clips',
                 clip_length: int = 10,
                 video_size: tuple[int, int] = (1920, 1080),
                 lowres_size: tuple[int, int] = (960, 544),
                 allowed_triggers: set[str] = {'person'}
                 ):
        super().__init__(daemon=True)
        self.storage_task_queue = storage_task_queue
        self.stop_event = stop_event
        self.db_path = db_path
        self.clip_dir = clip_dir
        self.clip_length = clip_length
        self.capture_buffer = capture_buffer
        self.video_size = video_size
        self.lowres_size = lowres_size
        self.detection_buffer = detection_buffer
        self.allowed_triggers = allowed_triggers
        self.clip_has_detections = False  # Track if the current clip has detections
        self.pending_start_task = None  # Store the pending start task if a clip is not yet started
        self.current_filename = None  # Track the current filename for the clip
        
        self.current_clip_start = 0.0
        self.current_clip_id = None
        self.detection_task_buffer = []
    def run(self):
        # Setup
        self.camera = Picamera2()
        self.storage_task_factory : TaskFactory = TaskFactory()

        # Configure picamera2
        video_config = self.camera.create_video_configuration(
            main={"size": self.video_size, "format": "RGB888"},
            lores={"size": self.lowres_size, "format": "RGB888"},
            encode="main",
        )

        self.camera.configure(video_config)

        # Limiting the FPS to reduce storage size.
        self.encoder = H264Encoder(bitrate=10000000, framerate=24)

        logger.info("Starting CaptureProcess")
        self.camera.start()
        logger.info("Camera started")
        self.current_clip_start = time.time() * 1000
        self.current_clip_id = uuid.uuid4().hex

        last_detection_version = -1  # Track the last processed detection version
        
        try:
            with metrics.time("capture_loop_time"):
                while not self.stop_event.is_set():
                    # Check if the current clip has exceeded the specified length, and if so, start a new clip.
                    with metrics.time("capture_clip_turnaround_time"):
                        if (time.time() * 1000) - self.current_clip_start > (self.clip_length * 1000):
                            self.current_clip_start = time.time() * 1000
                            self.end_clip()
                            self.start_clip()
                            self.detection_task_buffer = []  # Reset the detection task buffer for the new clip
                            self.last_detection_version = -1  # Reset the last detection version for the new clip
                    
                    with metrics.time("capture_frame_time"):
                        request = self.camera.capture_request()
                        try:
                            # Process the captured frame
                            lowres_frame = request.make_array("lores")

                            timestamp = request.get_metadata().get("SensorTimestamp")
                        finally:
                            request.release()

                    with metrics.time("buffer_write_time"):
                        # Write the low-resolution frame to the shared capture buffer
                        self.capture_buffer.write(lowres_frame, self.current_clip_id)

                    with metrics.time("detection_processing_time"):
                        detections, version = self.detection_buffer.get_with_version()

                        if version != last_detection_version:
                            last_detection_version = version
 
                            self.detection_task_buffer.append(detections) 

                            # Check if the current clip has detections and if any of them match the allowed triggers
                            if not self.clip_has_detections and detections:
                                if any(d.get("class_name") in self.allowed_triggers for d in detections):
                                    self.clip_has_detections = True

        except Exception as e:
            logger.error(f"Error in CaptureProcess: {e}", exc_info=True)
        finally:
            self.end_clip()  # Ensure the current clip is ended if the process is stopping
            self.camera.stop_encoder()
            self.camera.stop()
            self.camera.close()
            logger.info("CaptureProcess stopped")

    def start_clip(self):
        # Start a new clip by sending a start_clip task to the storage process
        self.current_clip_id = uuid.uuid4().hex  # Generate a unique ID for the new clip
        self.current_clip_start = time.time() * 1000  # Reset the clip start time
        self.clip_has_detections = False  # Reset the detection flag for the new clip

        os.makedirs(self.clip_dir, exist_ok=True)
        filename = os.path.join(self.clip_dir, f"clip_{int(time.time())}.mp4")
        self.current_filename = filename  # Store the current filename for the clip
        output = FfmpegOutput(filename)
        self.camera.start_encoder(self.encoder, output)

        logger.info(f"Started new clip {self.current_clip_id} at {self.current_clip_start}")

        self.pending_start_task = self.storage_task_factory.start_clip(
            clip_id=self.current_clip_id,
            start_time=self.current_clip_start,
            file_path=filename,
            trigger="continuous"
        )

    def end_clip(self):
        # End the current clip by sending an end_clip task to the storage process
        if self.current_clip_id is None:
            return  # No clip to end
        
        self.camera.stop_encoder()  # Stop the encoder, save the file, and finalize the clip

        end_time = time.time() * 1000
        
        if self.clip_has_detections:
            self.storage_task_queue.put(self.pending_start_task)  # type: ignore # Send the pending start task to storage
            end_clip_task = self.storage_task_factory.end_clip(
                clip_id=self.current_clip_id,
                ended_at=end_time
            )
            self.storage_task_queue.put(end_clip_task)  # Send the end clip task to storage
            logger.info(f"Ended clip {self.current_clip_id} at {end_time}")

            # Create an asynchronous task to process the detection tasks for this clip
            self.process_detection_tasks()
        else:
            logger.info(f"Clip {self.current_clip_id} ended without detections, not saving to storage.")
            try:
                if self.current_filename and os.path.exists(self.current_filename):
                    os.remove(self.current_filename)  # Remove the file if it exists
            except Exception as e:
                logger.error(f"Error removing file {self.current_filename}: {e}", exc_info=True)
            logger.info(f"Removed file {self.current_filename} for clip {self.current_clip_id} due to no detections.")
        
        self.pending_start_task = None  # Reset the pending start tasks
        self.current_filename = None  # Reset the current filename
        self.current_clip_id = None  # Reset the current clip ID

    def process_detection_tasks(self):
        # Process the detection tasks for the current clip
        if self.detection_task_buffer:
            for detection in self.detection_task_buffer:
                if detection:  # Only process if there are detections
                    for d in detection:
                        insert_detection_task = self.storage_task_factory.insert_detection(
                            clip_id=self.current_clip_id, # type: ignore
                            timestamp=d["timestamp"],
                            class_name=d["class_name"],
                            confidence=d["confidence"],
                            bbox_x=d["bbox_x"],
                            bbox_y=d["bbox_y"],
                            bbox_width=d["bbox_width"],
                            bbox_height=d["bbox_height"]
                        )
                        self.storage_task_queue.put(insert_detection_task)  # Send the detection task to storage
            logger.info(f"Processed {len(self.detection_task_buffer)} detection tasks for clip {self.current_clip_id}")