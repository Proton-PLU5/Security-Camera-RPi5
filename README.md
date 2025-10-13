# Raspberry Pi Security Camera (RPi5)

This small project captures frames from a Raspberry Pi camera and streams them
as MJPEG to a Flask webserver.

Prerequisites
- Raspberry Pi OS (for Picamera2 support) or any system with a webcam and OpenCV.
- Python 3.9+

Install dependencies

```powershell
python -m pip install -r requirements.txt
# On Raspberry Pi: also install system packages for libcamera/picamera2 if needed
# sudo apt update; sudo apt install -y python3-picamera2
```

Run

```powershell
python main.py
```

Open http://<raspberry-pi-ip>:5000 in your browser.

Notes
- For production use consider running behind gunicorn or another WSGI server.
- If Picamera2 is available it will be preferred. Otherwise OpenCV VideoCapture
  will be used as a fallback for development on other machines.
