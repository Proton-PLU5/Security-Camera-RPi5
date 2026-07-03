import logging

import numpy as np

from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from stream.frame_buffer import FrameBuffer
from stream.webrtc_stream import StreamThread
from ultralytics import YOLO

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s",
    )
    
    model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model
    # forces AutoBackend/NCNNBackend to actually initialize
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    model.predictor.model.net.opt.num_threads = 2 # type: ignore # Limit NCNN to 2 threads so other threads do not starve.
    
    storage_thread = StorageThread()
    storage_thread.start()

    detection_mailbox = MailBox()
    detection_store = DetectionStore()

    stream_buffer = FrameBuffer()

    capture_thread = CaptureThread(detection_mailbox=detection_mailbox, stream_buffer=stream_buffer, clip_dir="./clips", clip_length=10, storage_thread=storage_thread)
    capture_thread.start()

    detection_thread = DetectionThread(mailbox=detection_mailbox, detection_store=detection_store, model=model, storage_thread=storage_thread)
    detection_thread.start()

    stream_thread = StreamThread(buffer=stream_buffer, detection_store=detection_store)
    stream_thread.start()
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
    
    capture_thread.stop()
    capture_thread.join()
    detection_thread.stop()
    detection_thread.join()
    stream_thread.stop()
    stream_thread.join()
    storage_thread.stop()
    storage_thread.join()
