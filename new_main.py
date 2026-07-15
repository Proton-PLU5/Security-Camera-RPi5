import logging
from multiprocessing import Event, Lock, Queue, Manager, set_start_method
from firmware.config import Config
from mp.capture import CaptureProcess, CaptureBuffer
from mp.storage import StorageProcess
from mp.stream import StreamProcess
from mp.detect import DetectProcess, DetectionBuffer
from mp.vision_app import TextualLogHandler, VisionApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(levelname)s %(message)s")
set_start_method("spawn", force=True)  # Use 'spawn' to avoid issues with OpenCV and PyTorch in child processes

def main():
    storage_task_queue = Queue()
    stop_event = Event()
    lowres_size = (960, 544)  # Width, Height
    video_size = (1920, 1080)  # Width, Height
    capture_buffer = CaptureBuffer(shape = (lowres_size[1], lowres_size[0], 3)) # Height, Width, Channels
    detection_buffer = DetectionBuffer(max_bytes=65536)  # Adjust max_bytes as needed

    storage_process = StorageProcess(storage_task_queue, stop_event, db_path='storage.db')

    capture_process = CaptureProcess(
        storage_task_queue=storage_task_queue,
        stop_event=stop_event,
        db_path='storage.db',
        capture_buffer=capture_buffer,
        lowres_size=lowres_size,
        video_size=video_size,
        clip_dir='./clips',
    )

    detection_process = DetectProcess(
        stop_event=stop_event,
        capture_buffer=capture_buffer,
        storage_task_queue=storage_task_queue,
        detection_buffer=detection_buffer
    )

    stream_process = StreamProcess(
        buffer=capture_buffer,
        detection_buffer=detection_buffer,
        storage_db_path='storage.db',
        stop_event=stop_event)

    storage_process.start()
    capture_process.start()
    detection_process.start()
    stream_process.start()

    processes = {
        "capture": capture_process,
        "detection": detection_process,
        "stream": stream_process,
        "storage": storage_process
    }

    config = Config("config.toml")

    app = VisionApp(
        processes=processes
    )

    # Route stdlib logging into the app's log widget instead of real
    # stdout/stderr (which the full-screen app owns and would corrupt).
    # Set handlers directly on the root logger rather than
    # logging.basicConfig() - basicConfig() silently no-ops if anything
    # (ultralytics, picamera2, etc.) already attached a handler earlier.
    file_handler = logging.FileHandler(config.getString("log_file", "./terminal.log"), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(threadName)s] %(name)s: %(message)s")
    )

    tui_handler = TextualLogHandler(app)
    tui_handler.setLevel(logging.INFO)
    tui_handler.setFormatter(logging.Formatter("[%(threadName)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, tui_handler]
  
    try:
        app.run()
    finally:
        stop_event.set()  # Signal the storage process to stop
        storage_process.join()  # Wait for the storage process to finish
        capture_process.join()  # Wait for the capture process to finish
        detection_process.join()  # Wait for the detection process to finish
        stream_process.join()  # Wait for the stream process to finish
        capture_buffer.close()  # Close the shared buffer
        print(detection_process.exitcode)

if __name__ == "__main__":
    main()