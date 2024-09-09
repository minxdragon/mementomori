import serial
import time

def open_serial_connection(port, rate):
    try:
        return serial.Serial(port, rate)
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        exit(1)

def send_command(ser, command):
    #print(f'Sending: {command}')  # Uncomment for detailed logs
    ser.write((command + '\n').encode())
    response = ser.readline().strip().decode()
    #print(f'Response: {response}')  # Uncomment for detailed logs
    return response

def main():
    serial_port = '/dev/tty.usbserial-10'
    baud_rate = 115200
    gcode_file = 'output.gcode'

    ser = open_serial_connection(serial_port, baud_rate)

    try:
        # Initialize GRBL
        ser.write(b"\r\n\r\n")  # Wake up GRBL
        time.sleep(2)  # Wait for GRBL to initialize
        ser.flushInput()  # Flush startup text in serial input

        # Define and move to initial home position
        send_command(ser, 'G28')  # Alternatively, 'G92 X0 Y0' to set current position as home

        # Process G-code file
        try:
            with open(gcode_file, 'r') as f:
                for line in f:
                    send_command(ser, '$X')
                    l = line.strip()
                    if l:  # Ensure the line is not empty
                        response = send_command(ser, l)
                        if 'error' in response.lower():
                            send_command(ser, '$X')  # Clear any errors
        except FileNotFoundError:
            print(f"G-code file {gcode_file} not found.")
        except Exception as e:
            print(f"Error reading G-code file {gcode_file}: {e}")

        # Ensure to return to home position
        send_command(ser, 'G0 X0 Y0')  # Move back to home position

        time.sleep(1)  # Give some time for the last command to complete

    finally:
        ser.close()

if __name__ == "__main__":
    main()
