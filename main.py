import logging
import ncnn
from capture.recognition.face_queue import SnapshotAtomicQueue
from capture.recognition.face_recognition import FaceRecognition, FaceRecognitionThread
from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from stream.frame_buffer import FrameBuffer
from stream.webrtc_stream import StreamThread
from ultralytics import YOLO # type: ignore
from data.metrics import metrics
from app.vision_app import VisionApp, TextualLogHandler
from firmware.config import Config

def configure_ncnn():
    # NCNN THREAD LIMITING
    _orig_load_param = ncnn.Net.load_param # type: ignore

    def _load_param_with_thread_limit(self, path):
        self.opt.num_threads = 1  # set before weights get packed
        return _orig_load_param(self, path)

    ncnn.Net.load_param = _load_param_with_thread_limit # type: ignore

def main():    
    configure_ncnn()
    
    model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model

    config = Config("config.toml")
    data_base_path = config.getString("database_path", "storage.db")

    storage_thread = StorageThread(db_path=data_base_path)
    storage_thread.start()

    detection_mailbox = MailBox()
    detection_store = DetectionStore()
    stream_buffer = FrameBuffer()
    face_queue = SnapshotAtomicQueue()
    face_recognition = None
    face_mailbox = None

    capture_thread = CaptureThread(
        detection_mailbox=detection_mailbox, 
        stream_buffer=stream_buffer, 
        clip_dir=config.getString("clip_directory", "./clips"), 
        clip_length=config.getInt("clip_length", 10), 
        storage_thread=storage_thread,
        detection_store=detection_store
    )
    
    capture_thread.start()

    detection_thread = DetectionThread(
        mailbox=detection_mailbox, 
        detection_store=detection_store, 
        model=model, 
        storage_thread=storage_thread,
        face_queue=face_queue)
    detection_thread.start()

    face_recognition_thread = None

    if config.getBoolean("enable_face_recognition", True) == True:
        face_recognition = FaceRecognition(config)

        face_recognition_thread = FaceRecognitionThread(
            queue=face_queue,
            face_recognition=face_recognition,
            storage_thread=storage_thread,
            face_mailbox=face_mailbox
        )
        face_recognition_thread.start()

    stream_thread = StreamThread(
        buffer=stream_buffer,
        storage_db_path=data_base_path,
        detection_store=detection_store,
        config=config,
        host=config.getString("stream_host", "0.0.0.0"),
        port=config.getInt("stream_port", 8080),
        lowres_size=(config.getInt("stream_width", 960), config.getInt("stream_height", 540)),
        face_recognition=face_recognition,
        face_mailbox=face_mailbox
    )
    stream_thread.start()

    threads = {
        "capture": capture_thread,
        "detection": detection_thread,
        "stream": stream_thread,
        "storage": storage_thread
    }

    # Start the TUI application
    app = VisionApp(threads=threads)

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
        config.save_configurations()
        capture_thread.stop()
        capture_thread.join()
        detection_thread.stop()
        detection_thread.join()

        if face_recognition_thread is not None:
            face_queue.shutdown()
            face_recognition_thread.stop()
            face_recognition_thread.join()

        stream_thread.stop()
        stream_thread.join()
        storage_thread.stop()
        storage_thread.join()

        

if __name__ == "__main__":
    main()