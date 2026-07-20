import asyncio
import logging
import os
from typing import Tuple
from aiohttp import web
import sqlite3 as sqlite
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack
from multiprocessing import Process, Event
from data.metrics import metrics
from streaming.auth.authentication import Authenticator
from av import VideoFrame
import ssl
from capture.capture import CaptureBuffer
from capture.detect import DetectionBuffer

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: CaptureBuffer, lowres_size: Tuple[int, int] = (960, 540)):
        super().__init__()
        self.buffer = buffer
        self.lowres_size = lowres_size

    async def recv(self) -> VideoFrame:
        with metrics.time("stream_receive"):
            frame, clip_id = self.buffer.get()

            if frame is None:
                return await self.recv()

            video_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            video_frame.pts, video_frame.time_base = await self.next_timestamp()

            return video_frame

class StreamProcess(Process):
    def __init__(self, 
                 buffer: CaptureBuffer, 
                 detection_buffer: DetectionBuffer,
                 storage_db_path: str, 
                 stop_event: Event, # type: ignore
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 lowres_size: tuple[int, int] = (960, 544),
                ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.storage_db_path = storage_db_path
        self.host = host
        self.port = port
        self.lowres_size = lowres_size
        self.detection_buffer = detection_buffer
        self.stop_event = stop_event
        self.authenticator = Authenticator(storage_db_path)

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
        app.router.add_get("/clip/{clip_id}/detections", self.handle_get_detections_for_clip)
        app.router.add_get("/clip/{clip_id:.*}", self.handle_get_clip)
        app.router.add_get("/clips", self.handle_get_clips_before)
        app.router.add_get("/websocket/detections", self.handle_detection_websocket)
        app.router.add_post("/authenticate", self.handle_authentication)

        app.router.add_post("/offer", self.handle_offer)

        self.runner = web.AppRunner(app)
        await self.runner.setup()

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        site = web.TCPSite(self.runner, self.host, self.port, ssl_context=ssl_context)
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

    async def handle_get_detections_for_clip(self, request: web.Request) -> web.Response:

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.authenticator.validate_token(token) is None:
            return web.json_response({"error": "Invalid or missing token"}, status=401)

        clip_id = request.match_info.get("clip_id")

        connection = sqlite.connect(self.storage_db_path, timeout=5.0)
        result = {}
        try:
            cursor = connection.cursor()
            
            cursor.execute(
                "SELECT started_at FROM clips WHERE id = ?",
                (clip_id,),
            )

            clip_row = cursor.fetchone()

            if clip_row is None:
                return web.json_response({"error": "Clip not found"}, status=404)
            
            cursor.execute(
                "SELECT timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height"
                " FROM detections WHERE clip_id = ? ORDER BY timestamp ASC",
                (clip_id,),
            )

            detections = [
                {
                    "offset_seconds": (row[0] - clip_row[0]) / 1000.0,
                    "class_name": row[1],
                    "confidence": row[2],
                    "bbox_x": row[3],
                    "bbox_y": row[4],
                    "bbox_width": row[5],
                    "bbox_height": row[6],
                }
                for row in cursor.fetchall()
            ]

            result = {
                "lowres_size": list(self.lowres_size),
                "detections": detections,
            }
        finally:
            connection.close()

        return web.json_response(result)

    async def handle_get_clip(self, request: web.Request) -> web.StreamResponse:

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.authenticator.validate_token(token) is None:
            return web.json_response({"error": "Invalid or missing token"}, status=401)

        clip_id = request.match_info.get("clip_id")

        clip_path = f"./clips/clip_{clip_id}.mp4"

        if not os.path.exists(clip_path):
            return web.json_response(
                {"error": f"Clip with ID {clip_id} not found"},
                status=404,
            )

        return web.FileResponse(clip_path)
        

    def dict_factory(self, cursor: sqlite.Cursor, row):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    async def handle_get_clips_before(self, request: web.Request) -> web.Response:

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.authenticator.validate_token(token) is None:
            return web.json_response({"error": "Invalid or missing token"}, status=401)

        if request.query.get("before") is None:
            return web.json_response({"error": "Missing 'before' query parameter"}, status=400)

        # Epoch timestamp in seconds; default to 0 if not provided
        before_timestamp = float(request.query.get("before", 0)) * 1000  # Convert to milliseconds

        # Create a database connection and fetch clips that have ended after the given timestamp
        connection = sqlite.connect(self.storage_db_path)
        connection.row_factory = self.dict_factory
        clips = []

        try:
            cursor = connection.cursor()

            # Fetch clips from the database that have ended before the given timestamp
            cursor.execute("SELECT * FROM clips WHERE ended_at < ?", (before_timestamp,))
            clips = cursor.fetchall()
        except Exception as e:
            logging.error(f"Error fetching clips from database: {e}")
        finally:
            connection.close()

        return web.json_response(clips)

    async def handle_detection_websocket(self, request: web.Request) -> web.WebSocketResponse | web.Response:
        
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.authenticator.validate_token(token) is None:
            return web.json_response({"error": "Invalid or missing token"}, status=401)

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        last_version = -1
        try:
            while not self.stop_event.is_set() and not ws.closed:
                detections, version = self.detection_buffer.get_with_version()
                if version != last_version:
                    last_version = version
                    await ws.send_json(detections)
                await asyncio.sleep(0.1)
        except Exception as e:
            pass
        finally:
            await ws.close()

        return ws

    async def handle_offer(self, request: web.Request) -> web.Response:
        params = await request.json()

        # Check for valid token
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.authenticator.validate_token(token) is None:
            return web.json_response({"error": "Invalid or missing token"}, status=401)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]))
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                self.pcs.discard(pc)

        # Add the video track from the buffer
        video_track = CameraVideoTrack(buffer=self.buffer, lowres_size=self.lowres_size)
        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    
    async def handle_authentication(self, request: web.Request) -> web.Response:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return web.json_response({"error": "Username and password are required"}, status=400)

        is_authenticated = self.authenticator.authenticate(username, password)
        if not is_authenticated:
            return web.json_response({"error": "Invalid username or password"}, status=401)
        
        token = self.authenticator.generate_token(username)

        return web.json_response({"message": "Authentication successful", "token": token})

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse("mp/lite_viewer.html")