import logging

import numpy as np
import ncnn
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from stream.frame_buffer import FrameBuffer
from stream.webrtc_stream import StreamThread
from ultralytics import YOLO
from data.metrics import metrics
_orig_load_param = ncnn.Net.load_param # type: ignore

# NCNN THREAD LIMITING
def _load_param_with_thread_limit(self, path):
    self.opt.num_threads = 2  # set before weights get packed, not after
    return _orig_load_param(self, path)

ncnn.Net.load_param = _load_param_with_thread_limit # type: ignore

def build_commands(threads: dict) -> dict:
    def handle_metrics():
        metrics.report()

    def handle_status():
        status = {
            "capture_thread_alive": threads["capture"].is_alive(),
            "detection_thread_alive": threads["detection"].is_alive(),
            "stream_thread_alive": threads["stream"].is_alive(),
            "storage_thread_alive": threads["storage"].is_alive(),
        }
        return status
    
    def handle_help():
        print("Available commands: metrics, status, help, exit")

    return {
        "metrics": handle_metrics,
        "status": handle_status,
        "help": handle_help,
    }


def main():
    with patch_stdout():
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

        capture_thread = CaptureThread(
            detection_mailbox=detection_mailbox, 
            stream_buffer=stream_buffer, 
            clip_dir="./clips", 
            clip_length=10, 
            storage_thread=storage_thread)
        
        capture_thread.start()

        detection_thread = DetectionThread(
            mailbox=detection_mailbox, 
            detection_store=detection_store, 
            model=model, 
            storage_thread=storage_thread)
        detection_thread.start()

        stream_thread = StreamThread(
            buffer=stream_buffer, 
            detection_store=detection_store)
        stream_thread.start()

        threads = {
            "capture": capture_thread,
            "detection": detection_thread,
            "stream": stream_thread,
            "storage": storage_thread,
        }
        commands = build_commands(threads)

        session = PromptSession("> ")
        print("Type 'help' for a list of commands.")

        while True:
            try:
                user_input = session.prompt()
            except (EOFError, KeyboardInterrupt):
                break
 
            cmd = user_input.strip().lower()
            if not cmd:
                continue
            if cmd == "exit":
                break
            elif cmd in commands:
                commands[cmd]()
            else:
                print(f"Unknown command: {cmd!r}. Type 'help' for a list.")
        
        print("Shutting down threads...")
        capture_thread.stop()
        capture_thread.join()
        detection_thread.stop()
        detection_thread.join()
        stream_thread.stop()
        stream_thread.join()
        storage_thread.stop()
        storage_thread.join()

if __name__ == "__main__":
    main()