# Raspberry Pi Security Camera (RPi5)

This small project captures frames from a Raspberry Pi camera and streams them
as MJPEG to a Flask webserver.

Prerequisites
- Raspberry Pi OS (for Picamera2 support) or any system with a webcam and OpenCV.
- Python 3.9+

Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Run

```powershell
python main.py
```

Open http://<raspberry-pi-ip>:5000 in your browser.