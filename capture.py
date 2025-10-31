"""Camera capture helper.

Uses Picamera2 (libcamera) on Raspberry Pi 5 to capture JPEG frames and yield
them as bytes for MJPEG streaming. The camera runs continuously and multiple
clients can connect to view the same stream.

Default settings:
- Resolution: 1280x720 (720p HD)
- Framerate: 30 fps
- JPEG Quality: 95/100
"""

from typing import Generator, Tuple
import time
import io
import logging
import threading
import process

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
_stop_camera = False


def _camera_worker(resolution: Tuple[int, int] = (1280, 720), framerate: int = 30):
	"""Background thread that continuously captures frames from the camera."""
	global _camera, _current_frame, _stop_camera
	
	stream = io.BytesIO()
	
	try:
		logging.info("Initializing Picamera2 in background thread...")
		camera = Picamera2()
		
		# Configure for higher quality
		config = camera.create_video_configuration(
			main={'size': resolution, 'format': 'RGB888'},
			controls={'FrameRate': framerate},
			encode='main'
		)
		camera.configure(config)
		
		# Set JPEG quality via encoder (if available)
		try:
			camera.options['quality'] = 95
		except:
			pass
		
		camera.start()
		logging.info(f"Picamera2 started - {resolution[0]}x{resolution[1]} @ {framerate}fps - camera will run continuously")
		
		with _camera_lock:
			_camera = camera
		
		while not _stop_camera:
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


def start_camera(resolution: Tuple[int, int] = (1280, 720), framerate: int = 30):
	"""Start the camera in a background thread if not already running.
	
	Args:
		resolution: Video resolution (default 1280x720 for 720p HD)
		framerate: Frames per second (default 30)
	"""
	global _camera_thread, _stop_camera
	
	if _camera_thread is not None and _camera_thread.is_alive():
		logging.info("Camera already running")
		return
	
	_stop_camera = False
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


def stop_camera():
	"""Stop the camera thread gracefully."""
	global _camera, _camera_thread, _stop_camera
	
	logging.info("Stopping camera...")
	_stop_camera = True
	
	# Wait for camera thread to finish
	if _camera_thread is not None:
		_camera_thread.join(timeout=2)
		_camera_thread = None
	
	# Cleanup camera
	with _camera_lock:
		if _camera is not None:
			try:
				_camera.stop()
				_camera.close()
				logging.info("Camera stopped and cleaned up")
			except:
				pass
			_camera = None


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
				# Send the frame for processing and annotation
				# Make the frame into right format for processing
				# The utils.getInference function expects raw image bytes
				process.add_task(frame)

				# Retrieve the latest processed bounding boxes
				boxes = process.get_latest_bounding_boxes()

				# Draw boxes on the frame
				if boxes is not None:
					from PIL import Image, ImageDraw
					import numpy as np
					
					# Convert bytes to PIL Image
					image = Image.open(io.BytesIO(frame))
					draw = ImageDraw.Draw(image)
					
					# Draw each bounding box
					for box in boxes:
						x, y, w, h = box.tolist()
						left = x - w / 2
						top = y - h / 2
						right = x + w / 2
						bottom = y + h / 2
						draw.rectangle([left, top, right, bottom], outline="red", width=3)
					
					# Convert back to bytes
					buf = io.BytesIO()
					image.save(buf, format='JPEG')
					frame = buf.getvalue()

				# Prepare MJPEG frame
				yield frame
			
			# Small delay to avoid consuming too much CPU
			time.sleep(0.05)
			
	except GeneratorExit:
		logging.info("Client disconnected from video stream")
	except Exception as e:
		logging.error(f"Stream error: {e}")

