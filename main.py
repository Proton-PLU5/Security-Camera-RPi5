import logging
from multiprocessing import Event, Lock, Queue, Manager, set_start_method
import uuid
from data.config import Config
from capture.capture import CaptureProcess, CaptureBuffer
from data.storage import StorageProcess
from streaming.stream import StreamProcess
from capture.detect import DetectProcess, DetectionBuffer
from app.vision_app import TextualLogHandler, VisionApp
import socket
from zeroconf import ServiceInfo, Zeroconf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(levelname)s %(message)s")
set_start_method("spawn", force=True)  # Use 'spawn' to avoid issues with OpenCV and PyTorch in child processes

def main():
    storage_task_queue = Queue()
    stop_event = Event()
    lowres_size = (960, 544)  # Width, Height
    video_size = (1920, 1088)  # Width, Height
    capture_buffer = CaptureBuffer(shape = (lowres_size[1], lowres_size[0], 3)) # Height, Width, Channels
    detection_buffer = DetectionBuffer(max_bytes=65536)  # Adjust max_bytes as needed
    config = Config("config.toml")

    # Set up Zeroconf service for mDNS advertisement
    device_uuid = config.getString("device_uuid", uuid.uuid4().hex)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    camera_port = 8080

    service_type = "_camera._tcp.local."
    service_name = f"{hostname}.{service_type}"

    properties = {
        "id": device_uuid,
        "port": str(camera_port),
        "version": "1.0.0",
    }

    info = ServiceInfo(
        service_type,
        service_name,
        addresses=[socket.inet_aton(local_ip)],
        port=camera_port,
        properties=properties,
        server=f"{hostname}.local.",
    )

    zeroconf = Zeroconf()
    zeroconf.register_service(info)

    storage_process = StorageProcess(storage_task_queue, stop_event, db_path='storage.db')

    capture_process = CaptureProcess(
        storage_task_queue=storage_task_queue,
        stop_event=stop_event,
        db_path='storage.db',
        capture_buffer=capture_buffer,
        lowres_size=lowres_size,
        video_size=video_size,
        clip_dir='./clips',
        detection_buffer=detection_buffer,
    )

    detection_process = DetectProcess(
        stop_event=stop_event,
        capture_buffer=capture_buffer,
        storage_task_queue=storage_task_queue,
        detection_buffer=detection_buffer, 
        lowres_size=lowres_size
    )

    stream_process = StreamProcess(
        buffer=capture_buffer,
        detection_buffer=detection_buffer,
        storage_db_path='storage.db',
        stop_event=stop_event,
        port=camera_port,
        )

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

    """
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
    """
  
    try:
        while True:
            # Main loop can be used to monitor processes or handle other tasks
            for name, process in processes.items():
                if not process.is_alive():
                    logging.error(f"{name} process has stopped unexpectedly.")
                    stop_event.set()  # Signal all processes to stop
                    break
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping all processes...")
        stop_event.set()  # Signal all processes to stop
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()
        stop_event.set()  # Signal the storage process to stop
        storage_process.join()  # Wait for the storage process to finish
        capture_process.join()  # Wait for the capture process to finish
        detection_process.join()  # Wait for the detection process to finish
        stream_process.join()  # Wait for the stream process to finish
        capture_buffer.close()  # Close the shared buffer

        config.save_configurations()  # Save configurations to the config file

if __name__ == "__main__":
    main()
