from gpiozero import DigitalOutputDevice, PWMOutputDevice
from time import sleep

STANDBY = DigitalOutputDevice(17)
ain1 = PWMOutputDevice(27)
ain2 = PWMOutputDevice(22)

STANDBY.on()

speed = 0.2
  # 30% speed (0.0 to 1.0)

# Forward
ain1.value = speed
ain2.value = 0
sleep(0.3)
