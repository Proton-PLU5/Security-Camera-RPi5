import threading
import utils
import queue

# Thread-safe backlog for processing tasks
backlog = queue.Queue()
latest_result = None
running = True

def add_task(task):
    """Put a task into the backlog (thread-safe)."""
    backlog.put(task)

def get_task(block=True, timeout=None):
    """Get a task (or None if empty when timeout occurs)."""
    try:
        return backlog.get(block=block, timeout=timeout)
    except queue.Empty:
        return None

def task_done():
    """Signal that a retrieved task is complete."""
    backlog.task_done()

def stop_processing():
    # Stop the processing thread
    global running, processing_thread
    running = False
    processing_thread.join()

def process_backlog():
    global running
    while running:
        task = get_task()
        if task is None:
            break
        
        # Process the image passed.
        result = utils.getInference(task)
        whitelist = ["person", "bear"]
        boxes = utils.filterBoundingBoxes(result, whitelist)

        global latest_result
        latest_result = boxes

        task_done()

def get_latest_bounding_boxes():
    """Return the latest processed bounding boxes."""
    global latest_result
    return latest_result

processing_thread = threading.Thread(target=process_backlog)