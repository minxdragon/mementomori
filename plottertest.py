import serial
import time

# List of potential serial ports
serial_ports = [
    '/dev/tty.wchusbserial10',
    '/dev/tty.usbserial-10'
]

# Function to send a command and read the response
def send_command(ser, command):
    try:
        print(f"Sending command: {command}")
        ser.flushInput()
        ser.flushOutput()
        ser.write((command + '\n').encode())
        time.sleep(0.5)  # Wait for the response
        response = ser.readlines()
        return response
    except Exception as e:
        print(f"Error sending command {command}: {e}")
        return []

# Try each serial port
for port in serial_ports:
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(2)  # Wait for the connection to establish
        print(f"Connected to {port}")

        # Configure limit switches and initiate homing
        commands = [
            '$$',  # List current settings
            '$21=1',  # Enable hard limits
            '$22=1',  # Enable homing cycle
            '$23=3',  # Set homing direction invert mask (example value)
            '$24=25.000',  # Set homing locate feed rate (example value)
            '$25=500.000',  # Set homing search seek rate (example value)
            '$26=250',  # Set homing switch debounce delay (example value)
            '$27=1.000',  # Set homing switch pull-off distance (example value)
            '$X',  # Unlock GRBL
            '?',  # Get real-time status report
            '$H',  # Initiate homing cycle
        ]

        # Send commands and print responses
        for command in commands:
            response = send_command(ser, command)
            print(f"Response to {command}:")
            for line in response:
                print(line.decode().strip())

        # Close serial port
        ser.close()
        break  # Exit the loop if successful

    except Exception as e:
        print(f"Failed to connect to {port}: {e}")

print("Finished trying all ports.")