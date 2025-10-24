"""Entry point for the Raspberry Pi camera webserver."""
from server import run


if __name__ == '__main__':
	# Ask user if testing with Android emulator
	emulator_mode = input("Testing with Android emulator? (y/n): ").lower().strip() == 'y'
	
	emulator_ip = None
	if emulator_mode:
		emulator_ip = input("Enter your PC's IP address (where emulator is running): ").strip()
		print(f"\n✓ Emulator mode enabled")
		print(f"✓ Sending UDP packets to {emulator_ip}:12345")
		print(f"✓ Your emulator will receive these at 10.0.2.2:12345\n")
	else:
		print("\n✓ Broadcasting to all devices on the network\n")
	
	# Start the Flask development server with UDP discovery enabled.
	# The camera will broadcast its presence on UDP port 12345.
	# Customize camera_name if you have multiple cameras on the network.
	run(
		port=5000,
		camera_name="Raspberry Pi Camera",
		enable_discovery=True,
		emulator_mode=emulator_mode,
		emulator_ip=emulator_ip
	)