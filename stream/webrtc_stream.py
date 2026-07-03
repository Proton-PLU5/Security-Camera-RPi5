import asyncio
import logging
import threading
from typing import Optional, Tuple
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack
from av import VideoFrame
import cv2
import numpy as np

from capture.detection.detect import Detection, DetectionStore
from capture.mailbox import MailBox
from stream.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

def draw_detections(frame: np.ndarray, detection: Optional[Detection], lores_size: Tuple[int, int]) -> np.ndarray:
    """
    Draw the latest YOLO boxes onto a full-res frame.
    """
    if detection is None:
        return frame
 
    h, w = frame.shape[:2]
    lores_h, lores_w = detection.frame_size
    scale_x, scale_y = w / lores_w, h / lores_h
 
    for result in detection.boxes:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            class_name = str(class_id)
            names = getattr(result, "names", None)
            if isinstance(names, dict):
                class_name = names.get(class_id, class_name)
            elif isinstance(names, list) and 0 <= class_id < len(names):
                class_name = names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{class_name} {confidence:.2f}", (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
    return frame

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: FrameBuffer, detection_store: DetectionStore, lowres_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        self.buffer = buffer
        self.detection_store = detection_store
        self.lowres_size = lowres_size

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        frame = await asyncio.to_thread(self.buffer.get)
        if frame is None:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            frame = frame.copy()  # Copy the frame to avoid modifying the original

        detection = self.detection_store.latest()

        frame = draw_detections(frame, detection, self.lowres_size)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
    


class StreamThread(threading.Thread):
    def __init__(self, buffer: FrameBuffer, detection_store: DetectionStore, host: str = '0.0.0.0', port: int = 8080):
        super().__init__(daemon=True, name="StreamThread")
        self.buffer = buffer
        self.detection_store = detection_store
        self.stop_event = threading.Event()
        self.host = host
        self.port = port

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.runner: Optional[web.AppRunner] = None
        self.pcs: set[RTCPeerConnection] = set()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.serve())
        finally:
            self.loop.run_until_complete(self.shutdown())
            self.loop.close()

    def stop(self):
        self.stop_event.set()

    async def serve(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.handle_offer)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"WebRTC server started at http://{self.host}:{self.port}")

        # poll until stop_event is set
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

    async def handle_offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        ice_config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])

        pc = RTCPeerConnection(configuration=ice_config)
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            logger.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        # Add a video track that reads from the mailbox
        pc.addTrack(CameraVideoTrack(self.buffer, self.detection_store))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def shutdown(self):
        for pc in list(self.pcs):
            await pc.close()

        self.pcs.clear()
        if self.runner is not None:
            await self.runner.cleanup()

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse("stream/viewer.html")