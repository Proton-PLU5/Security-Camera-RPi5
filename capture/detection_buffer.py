import ctypes
import json
from multiprocessing import Array, Value
from data.metrics import metrics

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

    def get(self) -> list[dict]:
        # Return the latest detections snapshot
        active = self.active.value
        buf = self.buf_a if active == 0 else self.buf_b
        length_holder = self.len_a if active == 0 else self.len_b
        n = length_holder.value

        if n == 0: # No detections available
            return []
        
        payload = bytes(buf[:n]) # type: ignore
        return json.loads(payload.decode("utf-8"))

    def get_with_version(self) -> tuple[list[dict], int]:
        # Return the latest detections snapshot along with its version
        detections = self.get()
        version = self.version.value
        return detections, version

    def write(self, detections: list[dict]):
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
