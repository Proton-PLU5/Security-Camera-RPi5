import logging
from multiprocessing import shared_memory, Value
import uuid
import numpy as np

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

