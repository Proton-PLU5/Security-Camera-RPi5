# Raspberry Pi Security Camera (RPi5)

This project captures frames from a Raspberry Pi camera using Picamera2 and streams them as MJPEG to a Flask webserver. The camera also broadcasts UDP discovery packets so mobile apps can automatically find it on the network.

## Features
- MJPEG video streaming via HTTP
- UDP broadcast discovery (port 12345)
- Picamera2 support for Raspberry Pi 5
- Simple web interface

## Prerequisites
- Raspberry Pi OS (Bookworm or newer recommended for Picamera2 support)
- Python 3.9+
- Raspberry Pi Camera Module

## Install dependencies

On Raspberry Pi:
```bash
# Install system packages
sudo apt update
sudo apt install -y python3-picamera2 python3-pil

# Install Python dependencies
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

The server will:
- Start HTTP server on port 5000
- Broadcast UDP discovery packets on port 12345 every 2 seconds
- Log the camera IP and discovery info

## Access the camera

**Direct browser access:**
Open `http://<raspberry-pi-ip>:5000` in your browser.

**Via mobile app:**
Your mobile app will automatically discover the camera via UDP broadcast. The broadcast packet format:
```json
{
  "type": "security_camera",
  "name": "Raspberry Pi Camera",
  "ip": "192.168.1.100",
  "port": 5000
}
```

## Customization

Edit `main.py` to change:
- `port`: HTTP server port (default 5000)
- `camera_name`: Display name for discovery (default "Raspberry Pi Camera")
- `enable_discovery`: Enable/disable UDP broadcast (default True)

## Notes
- For production use consider running behind gunicorn or another WSGI server
- UDP broadcast uses port 12345 (ensure it's not blocked by firewall)
- Camera name should be unique if you have multiple cameras on the network