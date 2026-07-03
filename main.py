import logging

import numpy as np
import ncnn
from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from stream.frame_buffer import FrameBuffer
from stream.webrtc_stream import StreamThread
from ultralytics import YOLO

_orig_load_param = ncnn.Net.load_param

def _load_param_with_thread_limit(self, path):
    self.opt.num_threads = 2  # set before weights get packed, not after
    return _orig_load_param(self, path)

ncnn.Net.load_param = _load_param_with_thread_limit

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s",
    )
    
    model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model

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
