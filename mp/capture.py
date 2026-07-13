import logging
from multiprocessing import Event, Process, Queue, shared_memory, Value
import os
import time
import uuid
import numpy as np
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

    def get(self) -> np.ndarray:
        # Return the active buffer as a numpy array.
        active = self.active.value
        buf = self.shared_buffer_a.buf if active == 0 else self.shared_buffer_b.buf
        return np.ndarray(self.shape, dtype=self.dtype, buffer=buf).copy()
    
    def write(self, frame: np.ndarray):
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
                 db_path: str = 'storage.db',
                 clip_dir: str = 'clips',
                 clip_length: int = 10,
                 video_size: tuple[int, int] = (1920, 1080),
                 lowres_size: tuple[int, int] = (960, 540),
                 ):
        super().__init__(daemon=True)
        self.storage_task_queue = storage_task_queue
        self.storage_task_factory = TaskFactory()
        self.stop_event = stop_event
        self.db_path = db_path
        self.clip_dir = clip_dir
        self.clip_length = clip_length
        self.capture_buffer = capture_buffer
        self.camera = Picamera2()

        # Configure picamera2
        video_config = self.camera.create_video_configuration(
            main={"size": video_size, "format": "RGB888"},
            lores={"size": lowres_size, "format": "RGB888"},
            encode="main",
        )

        self.camera.configure(video_config)

        # Limiting the FPS to reduce storage size.
        self.encoder = H264Encoder(bitrate=10000000, framerate=24)

        self.current_clip_start = 0.0
        self.current_clip_id = None

    def run(self):
        logger.info("Starting CaptureProcess")
        self.camera.start()
        self.current_clip_start = time.time() * 1000
        self.start_clip()
        
        try:
            while not self.stop_event.is_set():
                # Check if the current clip has exceeded the specified length, and if so, start a new clip.
                if (time.time() * 1000) - self.current_clip_start > (self.clip_length * 1000):
                    self.current_clip_start = time.time() * 1000
                    self.end_clip()
                    self.start_clip()

                request = self.camera.capture_request()
                try:
                    # Process the captured frame
                    lowres_frame = request.make_array("lores")

                    timestamp = request.get_metadata().get("SensorTimestamp")
                finally:
                    request.release()

                # Write the low-resolution frame to the shared capture buffer
                self.capture_buffer.write(lowres_frame)
        except Exception as e:
            logger.error(f"Error in CaptureProcess: {e}")
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

        os.makedirs(self.clip_dir, exist_ok=True)
        filename = os.path.join(self.clip_dir, f"clip_{int(time.time())}.mp4")
        output = FfmpegOutput(filename)
        self.camera.start_encoder(self.encoder, output)

        logger.info(f"Started new clip {self.current_clip_id} at {self.current_clip_start}")

        start_clip_task = self.storage_task_factory.start_clip(
            clip_id=self.current_clip_id,
            start_time=self.current_clip_start,
            file_path=filename,
            trigger="continuous"
        )
        self.storage_task_queue.put(start_clip_task)

    def end_clip(self):
        # End the current clip by sending an end_clip task to the storage process
        if self.current_clip_id is None:
            return  # No clip to end
        
        self.camera.stop_encoder()  # Stop the encoder before ending the clip

        end_time = time.time() * 1000
        end_clip_task = self.storage_task_factory.end_clip(clip_id=self.current_clip_id, ended_at=end_time)
        self.storage_task_queue.put(end_clip_task)
        logger.info(f"Ended clip {self.current_clip_id} at {end_time}")