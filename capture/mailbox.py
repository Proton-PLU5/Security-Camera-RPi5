
import threading


class MailBox:
    """ A custom datastructure that stores the latest capture to handoff between threads. """
    def __init__(self):
        self.lock = threading.Lock()
        self.item = None
        self.has_item = threading.Event()

    def put(self, item):
        with self.lock:
            self.item = item
            self.has_item.set()

    def get(self, timeout=None):
        got = self.has_item.wait(timeout)

        if not got:
            return None

        with self.lock:
            item = self.item
            self.item = None
            self.has_item.clear()
            return item
        
    def empty(self) -> bool:
        with self.lock:
            return not self.has_item.is_set()