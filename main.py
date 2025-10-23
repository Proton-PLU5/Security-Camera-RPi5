"""Entry point for the Raspberry Pi camera webserver."""
from server import run


if __name__ == '__main__':
	# Start the Flask development server with UDP discovery enabled.
	# The camera will broadcast its presence on UDP port 12345.
	# Customize camera_name if you have multiple cameras on the network.
	run(
		port=5000,
		camera_name="Raspberry Pi Camera",
		enable_discovery=True
	)