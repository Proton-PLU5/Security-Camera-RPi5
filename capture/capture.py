import time
import cv2
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import logging
import threading
import os
from capture.mailbox import MailBox
from data.storage import StorageThread
from stream.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

class CaptureThread(threading.Thread):
    def __init__(self,
                 detection_mailbox: MailBox,
                 stream_buffer: FrameBuffer,
                 storage_thread: StorageThread,
                 video_size: tuple[int, int] = (1920, 1080),
                 lowres_size: tuple[int, int] = (640, 640),
                 clip_dir: str = "clips",
                 clip_length: int = 10,
                 ):
        
        super().__init__(daemon=True, name="CaptureThread")

        self.detection_mailbox = detection_mailbox
        self.stream_buffer = stream_buffer
        self.clip_dir = clip_dir
        self.clip_length = clip_length
        self.storage_thread = storage_thread
        self.camera = Picamera2()
        self.latest_clip_id = None

        video_config = self.camera.create_video_configuration(
            main={"size": video_size, "format": "RGB888"},
            lores={"size": lowres_size, "format": "YUV420"},
            encode="main",
        )
        self.camera.configure(video_config)

        self.encoder = H264Encoder(bitrate=10000000)
        self.stop_event = threading.Event()
        self.current_clip_start = 0.0

    def run(self):
        self.camera.start()
        self.latest_clip_id = self.start_new_clip()

        try:
            while not self.stop_event.is_set():
                if self.clip_length and (time.time() - self.current_clip_start > self.clip_length):
                    self.latest_clip_id = self.start_new_clip()
                
                request = self.camera.capture_request()
                try:
                    t0 = time.time()
                    lowres_frame = request.make_array("lores")
                    stream_frame = request.make_array("main")
                    t1 = time.time()

                    lowres_frame = cv2.cvtColor(lowres_frame, cv2.COLOR_YUV2RGB_I420)
                    stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_RGB2BGR)
                    t2 = time.time()

                    timestamp = request.get_metadata().get("SensorTimestamp")
                finally:
                    request.release()

                t3 = time.time()
                self.detection_mailbox.put((lowres_frame, timestamp, self.latest_clip_id))
                t4 = time.time()
                self.stream_buffer.put(stream_frame)
                t5 = time.time()

                logger.info(f"capture={t1-t0:.4f} convert={t2-t1:.4f} release={t3-t2:.4f} mailbox={t4-t3:.4f} streambuf={t5-t4:.4f}")
        except Exception as e:
            logger.error(f"Error in CaptureThread: {e}")
        finally:
            if self.latest_clip_id is not None:
                self.storage_thread.end_clip(self.latest_clip_id)
            self.camera.stop_encoder()
            self.camera.stop()
            self.camera.close()
            logger.info("CaptureThread stopped.")

    def stop(self):
        self.stop_event.set()

    def start_new_clip(self) -> str:
        os.makedirs(self.clip_dir, exist_ok=True)
        filename = os.path.join(self.clip_dir, f"clip_{int(time.time())}.mp4")
        output = FfmpegOutput(filename)

        if self.current_clip_start:
            self.camera.stop_encoder()

            if self.latest_clip_id is not None:
                self.storage_thread.end_clip(self.latest_clip_id)

        self.camera.start_encoder(self.encoder, output)

        self.current_clip_start = time.time()
        logger.info(f"Started new clip: {filename}")

        return self.storage_thread.start_clip(filename)
