import asyncio
import logging
from typing import Optional, Tuple
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack
from multiprocessing import Process, Queue, Event
from data.metrics import metrics

from av import VideoFrame
import numpy as np

from mp.capture import CaptureBuffer

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: CaptureBuffer, lowres_size: Tuple[int, int] = (960, 540)):
        super().__init__()
        self.buffer = buffer
        self.lowres_size = lowres_size

    async def recv(self) -> VideoFrame:
        with metrics.time("stream_receive"):
            frame = self.buffer.get()

            if frame is None:
                return await self.recv()

            # Convert the frame to a VideoFrame
            video_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            video_frame.pts = None  # Let aiortc handle the timestamp
            video_frame.time_base = None  # Let aiortc handle the time base

            return video_frame

class StreamProcess(Process):
    def __init__(self, 
                 buffer: CaptureBuffer, 
                 storage_db_path: str, 
                 stop_event: Event, # type: ignore
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 lowres_size: tuple[int, int] = (960, 540),
                ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.storage_db_path = storage_db_path
        self.host = host
        self.port = port
        self.lowres_size = lowres_size
        self.stop_event = stop_event

        self.pcs: set[RTCPeerConnection] = set()
    
    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.serve())
        finally:
            self.loop.run_until_complete(self.shutdown())
            self.loop.close()

    async def serve(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.handle_offer)

        self.runner = web.AppRunner(app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        logging.info(f"Stream server started at http://{self.host}:{self.port}")

        await self.loop.run_in_executor(None, self.stop_event.wait)

    async def shutdown(self):
        logging.info("Shutting down stream server...")
        for pc in self.pcs:
            await pc.close()
        
        self.pcs.clear()
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]))
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        # Add the video track from the buffer
        video_track = CameraVideoTrack(buffer=self.buffer, lowres_size=self.lowres_size)
        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse("mp/lite_viewer.html")