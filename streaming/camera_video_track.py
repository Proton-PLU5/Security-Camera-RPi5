
from typing import Tuple
from aiortc import VideoStreamTrack
from data.metrics import metrics
from av import VideoFrame
from capture.capture import CaptureBuffer

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: CaptureBuffer, lowres_size: Tuple[int, int] = (960, 540)):
        super().__init__()
        self.buffer = buffer
        self.lowres_size = lowres_size

    async def recv(self) -> VideoFrame:
        # aiortc calls this whenever it needs another frame. The timestamp
        # keeps playback in order for the remote viewer.
        with metrics.time("stream_receive"):
            frame, clip_id = self.buffer.get()

            if frame is None:
                return await self.recv()

            video_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            video_frame.pts, video_frame.time_base = await self.next_timestamp()

            return video_frame
