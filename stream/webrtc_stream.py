import asyncio
import logging
import threading
from typing import Optional, Tuple
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack
from av import VideoFrame
import cv2
from datetime import datetime, timezone
import numpy as np
from capture.detection.detect import Detection, DetectionStore
import sqlite3
from capture.mailbox import MailBox
from capture.recognition.face_recognition import FaceRecognition
from stream.frame_buffer import FrameBuffer
from pathlib import Path
from firmware.config import Config

logger = logging.getLogger(__name__)

def draw_detections(frame: np.ndarray, detection: Optional[Detection]) -> np.ndarray:
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # type: ignore
            cv2.putText( # type: ignore
                frame, f"{class_name} {confidence:.2f}", (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, # type: ignore
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

        frame = draw_detections(frame, detection)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
    
class StreamThread(threading.Thread):
    def __init__(self, buffer: FrameBuffer, 
                 detection_store: DetectionStore, 
                 storage_db_path: str, 
                 config: Config,
                 face_recognition: Optional[FaceRecognition] = None,
                 host: str = '0.0.0.0', 
                 port: int = 8080, 
                 lowres_size: Tuple[int, int] = (640, 640),
                 face_mailbox: Optional[MailBox] = None):
        super().__init__(daemon=True, name="StreamThread")
        self.buffer = buffer
        self.detection_store = detection_store
        self.storage_db_path = storage_db_path
        self.stop_event = threading.Event()
        self.host = host
        self.port = port
        self.lowres_size = lowres_size
        self.config = config
        self.face_recognition = face_recognition
        self.face_mailbox = face_mailbox

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
        app.router.add_post("/upload_face", self.handle_upload_face)

        app.router.add_get("/clips", self.handle_clips_list)
        app.router.add_get("/clips/find", self.handle_clip_find)
        app.router.add_get("/clips/{clip_id}/detections", self.handle_clip_detections)
        app.router.add_get("/clips/{clip_id}/recognitions", self.handle_clip_recognitions)
        app.router.add_get("/clips/get_latest_recognition", self.handle_get_latest_recognition)
        app.router.add_get("/clips/{clip_id}", self.handle_clip_file)
        

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"WebRTC server started at http://{self.host}:{self.port}")

        # poll until stop_event is set
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

    async def handle_upload_face(self, request: web.Request) -> web.Response:
        if self.face_recognition is None:
            return web.json_response({"error": "Face recognition is not enabled."}, status=400)

        img, name = None, None

        try:
            reader = await request.multipart()
            async for field in reader:
                if field.name == 'image':
                    image_data = await field.read(decode=True)
                    np_array = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                elif field.name == 'name':
                    name = await field.text()
        except Exception as e:
            logger.error(f"Error processing multipart data: {e}")
            return web.json_response({"error": "Invalid multipart data."}, status=400)
        
        if img is None or name is None:
            return web.json_response({"error": "Missing image or name."}, status=400)
        
        name = "".join(c for c in name if c.isalnum() or c == '_').strip()
        if not name:
            return web.json_response({"error": "Invalid name provided."}, status=400)
        
        try:
            await asyncio.to_thread(self.face_recognition.add_known_face, img, name)
        except Exception as e:
            logger.error(f"Error adding known face: {e}")
            return web.json_response({"error": "Failed to add known face."}, status=500)
        
        return web.json_response({"message": f"Face for '{name}' added successfully."})
            

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
    
    # --- Clip handling endpoints ---

    def query_clips_list(self):
        connection = sqlite3.connect(self.storage_db_path, timeout=5.0)
        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT id, started_at, ended_at, file_path, trigger "
                "FROM clips ORDER BY started_at DESC LIMIT 200"
            )
            return [
                {"id": r[0], "started_at": r[1], "ended_at": r[2], "file_path": r[3], "trigger": r[4]}
                for r in cursor.fetchall()
            ]
        finally:
            connection.close()

    def query_clip_at(self, timestamp: float):
        connection = sqlite3.connect(self.storage_db_path, timeout=5.0)
        try:
            cursor = connection.cursor()
                
            cursor.execute(
                "SELECT id, started_at, ended_at, file_path, trigger FROM clips "
                "WHERE started_at <= ? AND (ended_at IS NULL OR ended_at >= ?) "
                "ORDER BY started_at DESC LIMIT 1",
                (timestamp, timestamp),
            )
            row = cursor.fetchone()
            if row is not None:
                return {"id": row[0], "started_at": row[1], "ended_at": row[2], "file_path": row[3], "trigger": row[4]}
            return None
        finally:
            connection.close()

    def query_clip_by_id(self, clip_id: str):
        connection = sqlite3.connect(self.storage_db_path, timeout=5.0)
        try:
            cursor = connection.cursor()
            
            cursor.execute(
                "SELECT id, started_at, ended_at, file_path, trigger FROM clips WHERE id = ?",
                (clip_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                return {"id": row[0], "started_at": row[1], "ended_at": row[2], "file_path": row[3], "trigger": row[4]}
            return None
        finally:
            connection.close()

    def query_detections_for_clip(self, clip_id: str):
        connection = sqlite3.connect(self.storage_db_path, timeout=5.0)
        try:
            cursor = connection.cursor()
            
            cursor.execute(
                "SELECT started_at FROM clips WHERE id = ?",
                (clip_id,),
            )

            clip_row = cursor.fetchone()

            if clip_row is None:
                return None  # Clip not found
            
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

            return {
                "lowres_size": list(self.lowres_size),
                "detections": detections,
            }
        finally:
            connection.close()

    def query_recognitions_for_clip(self, clip_id: str):
        connection = sqlite3.connect(self.storage_db_path, timeout=5.0)
        try:
            cursor = connection.cursor()
            
            cursor.execute("SELECT started_at FROM clips WHERE id = ?", (clip_id,))
            clip_row = cursor.fetchone()

            if clip_row is None:
                return None  # Clip not found
            
            try:
                cursor.execute(
                    "SELECT timestamp, name FROM recognitions WHERE clip_id = ? ORDER BY timestamp ASC",
                    (clip_id,),
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # If the recognitions table doesn't exist yet, fallback to an empty list safely
                rows = []

            recognitions = [
                {
                    "offset_seconds": (row[0] - clip_row[0]) / 1000.0,
                    "name": row[1],
                }
                for row in rows
            ]

            return {
                "recognitions": recognitions,
            }
        finally:
            connection.close()

    async def handle_clips_list(self, request: web.Request) -> web.Response:
        clips = await asyncio.to_thread(self.query_clips_list)
        return web.json_response(clips)
    
    async def handle_clip_find(self, request: web.Request) -> web.Response:
        timestamp = request.query.get("timestamp")
        if not timestamp:
            return web.json_response({"error": "Missing timestamp parameter"}, status=400)
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_epoch = dt.timestamp() * 1000
        except ValueError:
            return web.json_response({"error": "Invalid timestamp format."}, status=400)

        clip = await asyncio.to_thread(self.query_clip_at, timestamp_epoch)
        if clip is None:
            return web.json_response({"error": "No clip found at the given timestamp"}, status=404)
        
        return web.json_response(clip)
    
    async def handle_clip_detections(self, request: web.Request) -> web.Response:
        clip_id = str(request.match_info.get("clip_id"))

        if not clip_id:
            return web.json_response({"error": "Missing clip_id parameter"}, status=400)

        detections = await asyncio.to_thread(self.query_detections_for_clip, clip_id)

        if detections is None:
            return web.json_response({"error": "Clip not found"}, status=404)

        return web.json_response(detections)
    
    async def handle_get_latest_recognition(self, request: web.Request) -> web.Response:
        if self.face_mailbox is None:
            return web.json_response({"error": "Face mailbox is not enabled."}, status=400)

        try:
            # If the queue is empty, this throws an exception (e.g., queue.Empty)
            latest_recognitions = await asyncio.to_thread(self.face_mailbox.get, timeout=0.1)
        except Exception:
            # Catching the exception prevents the 500 Internal Server Error crash
            latest_recognitions = None

        if latest_recognitions is None:
            return web.json_response({"error": "No recognitions found."}, status=404)

        return web.json_response(latest_recognitions)
    
    async def handle_clip_recognitions(self, request: web.Request) -> web.Response:
        clip_id = str(request.match_info.get("clip_id"))

        if not clip_id:
            return web.json_response({"error": "Missing clip_id parameter"}, status=400)

        recognitions = await asyncio.to_thread(self.query_recognitions_for_clip, clip_id)

        if recognitions is None:
            return web.json_response({"error": "Clip not found"}, status=404)

        return web.json_response(recognitions)
    
    async def handle_clip_file(self, request: web.Request) -> web.StreamResponse:
        clip_id = str(request.match_info.get("clip_id"))

        if not clip_id:
            return web.json_response({"error": "Missing clip_id parameter"}, status=400)

        clip = await asyncio.to_thread(self.query_clip_by_id, clip_id)

        if clip is None:
            return web.json_response({"error": "Clip not found"}, status=404)
        
        path = Path(clip["file_path"])
        if not path.exists():
            return web.json_response({"error": "File not found"}, status=404)
        
        return web.FileResponse(path)
    
    async def shutdown(self):
        for pc in list(self.pcs):
            await pc.close()

        self.pcs.clear()
        if self.runner is not None:
            await self.runner.cleanup()

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse("stream/viewer.html")