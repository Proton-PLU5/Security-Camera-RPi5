
import threading


class MailBox:
    """ A custom datastructure that stores the latest capture to handoff between threads. """
    def __init__(self):
        self.lock = threading.Lock()
        self.item = None
        self.semaphore = threading.BoundedSemaphore(0)

    def put(self, item):
        with self.lock:
            self.item = item
        
        self.semaphore.release() # signal that an item is available

    def get(self, timeout=None):
        acquired = self.semaphore.acquire(timeout=timeout)

        if not acquired:
            return None # timeout occurred, no item available

        with self.lock:
            item = self.item
            self.item = None
            return item
        
    def empty(self) -> bool:
        return self.item is None