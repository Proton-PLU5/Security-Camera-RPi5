import time
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    controls={"FrameDurationLimits": (17857, 17857)},
)
picam2.configure(config)
picam2.start()

n = 200
t0 = time.time()
for _ in range(n):
    picam2.capture_array("main")
elapsed = time.time() - t0
print(f"{n / elapsed:.1f} fps (raw capture, no lores, no processing)")
