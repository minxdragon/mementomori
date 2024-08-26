import serial
import time

# Open GRBL serial port
serial_port = '/dev/tty.usbserial-10'
baud_rate = 115200

try:
    s = serial.Serial(serial_port, baud_rate)
except Exception as e:
    print(f"Failed to open serial port {serial_port}: {e}")
    exit(1)

# Function to send a command to GRBL and print the response
def send_command(command):
    print(f'Sending: {command}')
    s.write((command + '\n').encode())
    grbl_out = s.readline()
    print(f' : {grbl_out.strip().decode()}')

# Function to query GRBL status
def query_status():
    print('Querying status...')
    s.write(b'?\n')
    grbl_out = s.readline()
    print(f'Status: {grbl_out.strip().decode()}')

# Initialize GRBL
try:
    # Wake up GRBL
    s.write(b"\r\n\r\n")
    time.sleep(2)  # Wait for GRBL to initialize
    s.flushInput()  # Flush startup text in serial input

    # Clear any previous errors
    send_command('~')  # Resume from error or pause state

    # Query GRBL status
    query_status()

    # Open G-code file
    gcode_file = 'output.gcode'
    try:
        with open(gcode_file, 'r') as f:
            # Stream G-code to GRBL
            for line in f:
                send_command('$X')  # Unlocks GRBL if it's in an alarm state
                l = line.strip()  # Strip all EOL characters for consistency
                send_command(l)

    except FileNotFoundError:
        print(f"G-code file {gcode_file} not found.")
    except Exception as e:
        print(f"Error reading G-code file {gcode_file}: {e}")

finally:
    # Close serial port
    s.close()
