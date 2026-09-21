# Raspberry Pi Security Camera (RPi5)

This project captures frames from a Raspberry Pi camera using Picamera2 and streams them as MJPEG to a Flask webserver. The camera also broadcasts UDP discovery packets so mobile apps can automatically find it on the network.

https://github.com/user-attachments/assets/f5a943ba-e7e2-4561-83d6-b6f399645a2e

## Features
- MJPEG video streaming via HTTP
- Continuous camera operation
- Multiple simultaneous viewers supported
- UDP broadcast discovery (port 12345)
- Picamera2 support for Raspberry Pi 5

## Prerequisites
- Raspberry Pi OS (Bookworm or newer recommended for Picamera2 support)
- Python 3.9+
- Raspberry Pi Camera Module

## Install dependencies

On Raspberry Pi:
```bash
# Install system packages
sudo apt update
sudo apt install -y uv
uv sync
```

## Run

```bash
launch.bash
```

The server will:
- Start HTTP server on port 5000
- Broadcast/send UDP discovery packets for mobile devices to connect
- Log the camera IP and discovery info

## Access the camera

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

## Future Plans
- Implement UDP Hole Punching to connect to the camera from any network
- Create a backend service to handle user registration, account management and proper authentication setups.
- Implement servo controls using GPIO Pins and create custom endpoints for camera movement
- Implement camere partrol features and other automation settings.

App Repo: https://github.com/Proton-PLU5/Security-Camera-App
