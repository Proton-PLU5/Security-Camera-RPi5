"""Camera capture helper.

This module uses the legacy ``picamera`` (PiCamera) library to access the
Raspberry Pi camera and yields JPEG-encoded frames as bytes. If ``picamera``
is not available the code falls back to OpenCV's VideoCapture (useful for
development on non-Pi machines).

Public API:
- get_frame() -> generator yielding JPEG bytes
"""

from typing import Generator, Tuple
import time
try:
	from picamera import PiCamera
	from picamera.array import PiRGBArray
	_HAS_PICAMERA = True
except Exception:
	_HAS_PICAMERA = False

try:
	import cv2
	_HAS_OPENCV = True
except Exception:
	_HAS_OPENCV = False


def _picamera_generator(resolution: Tuple[int, int] = (640, 480), framerate: int = 20) -> Generator[bytes, None, None]:
	if not _HAS_PICAMERA:
		raise RuntimeError('picamera library is not available')

	camera = PiCamera()
	camera.resolution = resolution
	camera.framerate = framerate
	raw = PiRGBArray(camera, size=resolution)

	# give camera time to warm up
	time.sleep(0.1)

	try:
		for frame in camera.capture_continuous(raw, format='bgr', use_video_port=True):
			image = frame.array
			if _HAS_OPENCV:
				ret, jpeg = cv2.imencode('.jpg', image)
				if ret:
					yield jpeg.tobytes()
			else:
				# If OpenCV isn't present, use PIL to encode (try to import lazily)
				try:
					from PIL import Image
					import io
					img = Image.fromarray(image)
					buf = io.BytesIO()
					img.save(buf, format='JPEG')
					yield buf.getvalue()
				except Exception:
					# cannot encode; skip frame
					pass

			raw.truncate(0)
			raw.seek(0)
			time.sleep(1.0 / framerate)
	finally:
		camera.close()


def _opencv_generator(device: int = 0, width: int = 640, height: int = 480, framerate: int = 20) -> Generator[bytes, None, None]:
	if not _HAS_OPENCV:
		raise RuntimeError('OpenCV is not available')

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
	"""Return a generator yielding JPEG frames as bytes.

	Preference order:
	1. picamera (PiCamera)
	2. OpenCV VideoCapture

	Raises RuntimeError if no backend is available.
	"""
	if _HAS_PICAMERA:
		return _picamera_generator()
	if _HAS_OPENCV:
		return _opencv_generator()
	raise RuntimeError('No camera backend available: install picamera or opencv-python')

