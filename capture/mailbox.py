
import threading


class MailBox:
    """ A custom datastructure that stores the latest capture to handoff between threads. """
    def __init__(self):
        self.lock = threading.Lock()
        self.item = None
        self.event = threading.Event()

    def put(self, item):
        with self.lock:
            self.item = item
            self.event.set()

    def get(self, timeout=None):
        acquired = self.event.wait(timeout)

        if not acquired:
            return None # timeout occurred, no item available

        with self.lock:
            item = self.item
            self.item = None
            self.event.clear()
            return item
        
    def empty(self) -> bool:
        return not self.event.is_set()