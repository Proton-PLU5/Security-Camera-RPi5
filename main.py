"""Entry point for the Raspberry Pi camera webserver."""
from server import run
import capture
import process
import atexit
import signal
import sys


def cleanup():
	"""Cleanup function to stop all threads gracefully."""
	print("\n\nShutting down...")
	process.stop_processing()
	capture.stop_camera()
	print("Cleanup complete. Goodbye!")


def signal_handler(sig, frame):
	"""Handle Ctrl+C and other termination signals."""
	cleanup()
	sys.exit(0)


if __name__ == '__main__':
	# Register cleanup handlers
	atexit.register(cleanup)
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)
	
	print("Starting Raspberry Pi Camera Server...")
	print("Broadcasting to all devices on the network\n")
	
	# Start the Flask development server with UDP discovery enabled.
	# The camera will broadcast its presence on UDP port 12345.
	run(
		port=5000,
		camera_name="Raspberry Pi Camera",
		enable_discovery=True
	)