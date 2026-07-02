from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./capture/detection/yolo26s.pt")  # Load the YOLO model

    storage_thread = StorageThread()
    storage_thread.start()

    mailbox = MailBox()
    detection_store = DetectionStore()

    capture_thread = CaptureThread(mailbox=mailbox, clip_dir="./clips", clip_length=10, storage_thread=storage_thread)
    capture_thread.start()

    detection_thread = DetectionThread(mailbox=mailbox, detection_store=detection_store, model=model, storage_thread=storage_thread)
    detection_thread.start()
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
    
    capture_thread.stop()
    capture_thread.join()
    detection_thread.stop()
    detection_thread.join()
    storage_thread.stop()
    storage_thread.join()