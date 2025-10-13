"""Entry point for the Raspberry Pi camera webserver."""
from server import run


if __name__ == '__main__':
	# Start the Flask development server. On the Pi you may want to run
	# behind a production WSGI server or systemd service.
	run()