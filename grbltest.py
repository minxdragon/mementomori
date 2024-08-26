import serial
import time

# Open grbl serial port
serial_port = '/dev/tty.usbserial-10'
baud_rate = 115200

try:
    s = serial.Serial(serial_port, baud_rate)
except Exception as e:
    print(f"Failed to open serial port {serial_port}: {e}")
    exit(1)

# Open g-code file
gcode_file = 'output.gcode'
try:
    with open(gcode_file, 'r') as f:
        # Wake up grbl
        s.write(b"\r\n\r\n")
        time.sleep(2)  # Wait for grbl to initialize
        s.flushInput()  # Flush startup text in serial input

        # Stream g-code to grbl
        for line in f:
            l = line.strip()  # Strip all EOL characters for consistency
            print(f'Sending: {l}')
            s.write((l + '\n').encode())  # Send g-code block to grbl
            grbl_out = s.readline()  # Wait for grbl response with carriage return
            print(f' : {grbl_out.strip().decode()}')

except FileNotFoundError:
    print(f"G-code file {gcode_file} not found.")
except Exception as e:
    print(f"Error reading g-code file {gcode_file}: {e}")

# Close serial port
s.close()