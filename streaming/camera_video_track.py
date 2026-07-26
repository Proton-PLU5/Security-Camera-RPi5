
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

    '''
        this method is called by the aiortc library to get the next video frame to send over the WebRTC connection. 
        It retrieves a frame from the CaptureBuffer, converts it to a VideoFrame object, and sets the presentation timestamp (pts) 
        and time base for synchronization.
    '''
    async def recv(self) -> VideoFrame:
        with metrics.time("stream_receive"):
            frame, clip_id = self.buffer.get()

            if frame is None:
                return await self.recv()

            video_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            video_frame.pts, video_frame.time_base = await self.next_timestamp()

            return video_frame
