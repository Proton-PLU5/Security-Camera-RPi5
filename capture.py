"""Camera capture helper.

Uses Picamera2 (libcamera) on Raspberry Pi 5 to capture JPEG frames and yield
them as bytes for MJPEG streaming. The camera runs continuously and multiple
clients can connect to view the same stream.
"""

from typing import Generator, Tuple
import time
import io
import logging
import threading

# Picamera2 (required)
try:
	from picamera2 import Picamera2
	_HAS_PICAMERA2 = True
except Exception:
	_HAS_PICAMERA2 = False

logging.basicConfig(level=logging.INFO)

# Global camera instance and frame buffer
_camera = None
_camera_lock = threading.Lock()
_current_frame = None
_frame_lock = threading.Lock()
_camera_thread = None


def _camera_worker(resolution: Tuple[int, int] = (640, 480), framerate: int = 20):
	"""Background thread that continuously captures frames from the camera."""
	global _camera, _current_frame
	
	stream = io.BytesIO()
	
	try:
		logging.info("Initializing Picamera2 in background thread...")
		camera = Picamera2()
		camera.configure(camera.create_preview_configuration({'size': resolution}))
		camera.start()
		logging.info("Picamera2 started successfully - camera will run continuously")
		
		with _camera_lock:
			_camera = camera
		
		while True:
			# Capture frame
			camera.capture_file(stream, format='jpeg')
			frame_data = stream.getvalue()
			
			# Update the shared frame buffer
			with _frame_lock:
				_current_frame = frame_data
			
			stream.seek(0)
			stream.truncate()
			time.sleep(1.0 / framerate)
			
	except Exception as e:
		logging.error(f"Camera worker error: {e}")
	finally:
		with _camera_lock:
			if _camera is not None:
				try:
					_camera.stop()
					_camera.close()
					logging.info("Camera stopped")
				except:
					pass
				_camera = None


def start_camera(resolution: Tuple[int, int] = (640, 480), framerate: int = 20):
	"""Start the camera in a background thread if not already running."""
	global _camera_thread
	
	if _camera_thread is not None and _camera_thread.is_alive():
		logging.info("Camera already running")
		return
	
	_camera_thread = threading.Thread(
		target=_camera_worker,
		args=(resolution, framerate),
		daemon=True
	)
	_camera_thread.start()
	
	# Wait for camera to initialize
	timeout = 5
	start_time = time.time()
	while _camera is None and (time.time() - start_time) < timeout:
		time.sleep(0.1)
	
	if _camera is None:
		raise RuntimeError("Failed to initialize camera")


def get_frame() -> Generator[bytes, None, None]:
	"""Return a generator yielding JPEG frames as bytes.
	
	Multiple clients can call this simultaneously - they all receive
	the same frames from the continuously running camera.
	"""
	if not _HAS_PICAMERA2:
		raise RuntimeError('Picamera2 is not available. Install picamera2 on the Raspberry Pi.')
	
	# Ensure camera is started
	start_camera()
	
	logging.info("Client connected to video stream")
	
	try:
		while True:
			# Get the current frame
			with _frame_lock:
				frame = _current_frame
			
			if frame is not None:
				yield frame
			
			# Small delay to avoid consuming too much CPU
			time.sleep(0.05)
			
	except GeneratorExit:
		logging.info("Client disconnected from video stream")
	except Exception as e:
		logging.error(f"Stream error: {e}")

