import collections
import threading

"""
    An atomic queue that holds a single snapshot of data for processing,
    prevents new data from being added until the current snapshot is processed.

    When accessing items from the queue, if the queue is empty, the thread will block until a new snapshot arrives.
    Preventing busy-waiting and allowing for more efficient processing of data.
"""
class SnapshotAtomicQueue:
    def __init__(self):
        self.queue = collections.deque()
        self.condition = threading.Condition()
        self.is_locked = False
        self.is_shutdown = False
 
    def add(self, item) -> bool:
        """
        Add an item to the current batch. Returns False (item NOT
        added) if the queue is currently locked - i.e. the previous
        frame's batch hasn't finished draining yet, so this frame's
        faces are skipped entirely rather than partially admitted.
        """
        with self.condition:
            if self.is_locked or self.is_shutdown:
                return False
            self.queue.append(item)
            self.condition.notify()
            return True
 
    def lock(self):
        """
        Seal the current batch so the next frame's add() calls are
        refused until this batch is fully drained. Only actually locks
        if there's something left in the queue to protect - locking an
        empty queue would have nothing to trigger the auto-unlock in
        get(), leaving it locked forever.
        """
        with self.condition:
            if self.queue:
                self.is_locked = True
 
    def get(self):
        """
        Pop the next item, FIFO. Blocks if empty until an item arrives
        or shutdown() is called. Automatically unlocks once this pop
        empties the queue, admitting the next frame's batch.
        """
        with self.condition:
            while not self.queue and not self.is_shutdown:
                self.condition.wait()
 
            if self.is_shutdown and not self.queue:
                return None
 
            item = self.queue.popleft()
 
            if not self.queue:
                self.is_locked = False  # batch fully drained - next frame can be admitted
 
            return item
 
    def shutdown(self):
        """Wake any waiting get() call and signal it to stop waiting."""
        with self.condition:
            self.is_shutdown = True
            self.condition.notify_all()