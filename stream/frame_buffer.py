import threading

class FrameBuffer:
    """Always holds the latest frame (no consumption)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.item = None

    def put(self, item):
        with self.lock:
            self.item = item  # overwrite old frame

    def get(self):
        with self.lock:
            return self.item

    def empty(self):
        with self.lock:
            return self.item is None