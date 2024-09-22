import serial
import time

# List of common baud rates to test
baud_rates = [9600, 19200, 38400, 57600, 115200, 250000]

# Function to send a command and read the response
def send_command(ser, command):
    ser.write((command + '\n').encode())
    time.sleep(0.5)  # Wait for the response
    response = ser.readlines()
    return response

# Test each baud rate
for baud_rate in baud_rates:
    try:
        print(f"Testing baud rate: {baud_rate}")
        ser = serial.Serial('/dev/tty.usbserial-10', baud_rate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish

        # Send a test command
        response = send_command(ser, '$')
        print(f"Response at {baud_rate}:")
        for line in response:
            print(line.decode().strip())

        ser.close()
    except Exception as e:
        print(f"Failed to communicate at baud rate {baud_rate}: {e}")