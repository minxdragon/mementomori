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
    response = grbl_out.strip().decode()
    print(f' : {response}')
    return response

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

    # Define home location using G92 (set position to zero)
    send_command('G92 X0 Y0')  # Sets the current position as (0,0)

    # Draw a basic square
    square_commands = [
        'G00 X0 Y0',  # Move to the starting corner
        'G01 X50 Y0',  # Draw the first side
        'G01 X50 Y50',  # Draw the second side
        'G01 X0 Y50',  # Draw the third side
        'G01 X0 Y0'   # Draw the fourth side to close the square
    ]

    for command in square_commands:
        # Set feed rate inside the loop if necessary
        send_command('F1000')  # Set the feed rate to 1000 mm/min

        # Send the G-code command
        response = send_command(command)
        
        # Error handling
        if 'error' in response.lower():
            print("Error detected, resetting GRBL...")
            send_command('$X')  # Unlock GRBL if it's in an alarm state

finally:
    # Close serial port
    s.close()
