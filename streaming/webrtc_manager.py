import logging
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from streaming.camera_video_track import CameraVideoTrack


class WebRTCManager:
    def __init__(self, buffer, lowres_size):
        self.buffer = buffer
        self.lowres_size = lowres_size
        self.pcs: set[RTCPeerConnection] = set()

    async def create_answer(self, sdp : str, type : str) -> RTCSessionDescription:
        offer = RTCSessionDescription(sdp=sdp, type=type)
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
        )
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                self.pcs.discard(pc)

        video_track = CameraVideoTrack(buffer=self.buffer, lowres_size=self.lowres_size)
        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc.localDescription

    