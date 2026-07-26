import asyncio
import logging
import os
from aiohttp import web
import sqlite3 as sqlite
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from multiprocessing import Process, Event, Value
from streaming.auth.authentication import Authenticator
from capture.capture import CaptureBuffer
from capture.detect import DetectionBuffer
from streaming.camera_video_track import CameraVideoTrack
from streaming.clips_repository import ClipsRepository
from streaming.middlewares import auth_middleware
from streaming.routes import clips, detections, pairing, webrtc
from streaming.webrtc_manager import WebRTCManager

class StreamProcess(Process):
    def __init__(self, 
                 buffer: CaptureBuffer, 
                 detection_buffer: DetectionBuffer,
                 storage_db_path: str, 
                 stop_event: Event, # type: ignore
                 pairing_mode: Value, # type: ignore
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 lowres_size: tuple[int, int] = (960, 544),
                 clips_dir: str = "./clips"
                ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.storage_db_path = storage_db_path
        self.host = host
        self.port = port
        self.lowres_size = lowres_size
        self.detection_buffer = detection_buffer
        self.stop_event = stop_event
        self.pairing_mode = pairing_mode
        self.clips_dir = clips_dir

        self.pcs: set[RTCPeerConnection] = set()
    
    def run(self):
        self.authenticator = Authenticator(self.storage_db_path)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.serve())
        finally:
            self.loop.run_until_complete(self.shutdown())
            self.loop.close()

    async def serve(self):
        app = web.Application(middlewares=[auth_middleware])

        app["authenticator"] = Authenticator(self.storage_db_path)
        app["webrtc_manager"] = WebRTCManager(self.buffer, self.lowres_size)
        app["clips_repo"] = ClipsRepository(self.storage_db_path)
        app["detection_buffer"] = self.detection_buffer
        app["pairing_mode"] = self.pairing_mode
        app["stop_event"] = self.stop_event
        app["lowres_size"] = self.lowres_size
        app["clips_dir"] = self.clips_dir
        
        app.add_routes(clips.routes)
        app.add_routes(detections.routes)
        app.add_routes(pairing.routes)
        app.add_routes(webrtc.routes)
        app.router.add_get("/", self.index)

        self.runner = web.AppRunner(app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        logging.info(f"Stream server started at http://{self.host}:{self.port}")
        await self.loop.run_in_executor(None, self.stop_event.wait)

        self.app = app

    async def shutdown(self):
        logging.info("Shutting down stream server...")
        await self.app["webrtc_manager"].close_all()
        if hasattr(self, "runner"):
            await self.runner.cleanup()

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse("mp/lite_viewer.html")