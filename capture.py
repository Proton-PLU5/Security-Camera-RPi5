"""Camera capture helper.

Uses Picamera2 (libcamera) on Raspberry Pi 5 to capture JPEG frames and yield
them as bytes for MJPEG streaming. This file intentionally does not use the
legacy `picamera` or OpenCV backends.
"""

from typing import Generator, Tuple
import time
import io

# Picamera2 (required)
try:
	from picamera2 import Picamera2
	_HAS_PICAMERA2 = True
except Exception:
	_HAS_PICAMERA2 = False


def _picamera2_generator(resolution: Tuple[int, int] = (640, 480), framerate: int = 20) -> Generator[bytes, None, None]:
	if not _HAS_PICAMERA2:
		raise RuntimeError('Picamera2 is not available')

	picam = Picamera2()
	picam.configure(picam.create_preview_configuration({'size': resolution}))
	picam.start()
	stream = io.BytesIO()
	try:
		while True:
			# capture_file writes a JPEG to the provided file-like object
			picam.capture_file(stream, format='jpeg')
			data = stream.getvalue()
			if data:
				yield data
			stream.seek(0)
			stream.truncate()
			time.sleep(1.0 / framerate)
	finally:
		picam.stop()


# intentionally Picamera2-only implementation (no legacy picamera or OpenCV)


def get_frame() -> Generator[bytes, None, None]:
	"""Return a generator yielding JPEG frames as bytes using Picamera2.

	This module requires Picamera2 (libcamera). If Picamera2 isn't
	available a RuntimeError will be raised.
	"""
	if _HAS_PICAMERA2:
		return _picamera2_generator()
	raise RuntimeError('Picamera2 is not available. Install picamera2 on the Raspberry Pi.')

