import threading
import utils
import queue
import logging

# Thread-safe backlog for processing tasks
backlog = queue.Queue(maxsize=1)  # Only allow 1 frame in queue at a time
latest_result = None
running = True
is_processing = False  # Flag to track if currently processing

logging.basicConfig(level=logging.INFO)

def add_task(task):
    """Put a task into the backlog (thread-safe). Only adds if not currently processing."""
    global is_processing
    if not is_processing:
        try:
            backlog.put_nowait(task)  # Non-blocking put
        except queue.Full:
            pass  # Skip this frame if queue is full

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
    """Stop the processing thread gracefully."""
    global running, processing_thread
    running = False
    # Add a sentinel value to unblock the queue.get() if it's waiting
    backlog.put(None)
    # Wait for the thread to finish
    if processing_thread is not None and processing_thread.is_alive():
        processing_thread.join(timeout=2)

def process_backlog():
    """Process tasks from the backlog queue."""
    global running, latest_result, is_processing
    logging.info("Processing thread started")
    
    # Configure which classes to detect
    whitelist = ["person", "dog", "cat", "bear", "bird"]  # Add or remove classes as needed
    
    while running:
        task = get_task()
        if task is None:
            break
        
        is_processing = True
        
        # Process the image passed.
        result = utils.getInference(task)
        
        # Filter the result to only show whitelisted classes
        if len(whitelist) > 0:
            result = utils.filterResult(result, whitelist)
        
        # Store the filtered result object
        latest_result = result
        
        num_detections = len(result.boxes) if result.boxes is not None else 0
        if num_detections > 0:
            detected_classes = [result.names[int(cls)] for cls in result.boxes.cls]
            logging.info(f"Detected {num_detections} objects: {detected_classes}")
        else:
            logging.debug(f"No objects detected")

        task_done()
        is_processing = False
        
    logging.info("Processing thread stopped")

def get_latest_bounding_boxes():
    """Return the latest processed result object."""
    global latest_result
    return latest_result

def is_busy():
    """Check if currently processing a frame."""
    global is_processing
    return is_processing

def start_processing():
    """Start the processing thread."""
    global processing_thread
    processing_thread = threading.Thread(target=process_backlog, daemon=True)
    processing_thread.start()

# Start the processing thread on module import
start_processing()