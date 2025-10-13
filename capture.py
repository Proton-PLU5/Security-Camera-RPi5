"""Camera capture helper.

Provides a get_frame() generator that yields JPEG-compressed frames as bytes.
Uses Picamera2 on Raspberry Pi OS when available, otherwise falls back to
OpenCV's VideoCapture (useful for development on non-RPi machines).
"""
from typing import Generator
import time
import cv2

def _opencv_generator(device=0, width=640, height=480, framerate=20) -> Generator[bytes, None, None]:
	cap = cv2.VideoCapture(device)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				time.sleep(0.1)
				continue
			ret, jpeg = cv2.imencode('.jpg', frame)
			if not ret:
				continue
			yield jpeg.tobytes()
			time.sleep(1.0 / framerate)
	finally:
		cap.release()


def get_frame() -> Generator[bytes, None, None]:
	"""Return a generator that yields JPEG frames as bytes.
	"""
	return _opencv_generator()

