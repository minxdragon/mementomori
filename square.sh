#!/bin/bash

# Define the serial port and baud rate
SERIAL_PORT="/dev/tty.usbserial-10"
BAUD_RATE=115200

# Configure the serial port
echo "Configuring serial port..."
stty -f $SERIAL_PORT $BAUD_RATE cs8 -cstopb -parenb raw -echo

# Function to send G-code commands with a delay
send_gcode() {
    echo -e "$1" > $SERIAL_PORT
    if [ $? -ne 0 ]; then
        echo "Error sending command: $1" >&2
        exit 1
    fi
    echo "Sent: $1"
    sleep 0.5  # Adjust the delay if necessary
}

echo "Initializing plotter..."
# Send initialization commands
send_gcode "G21"    # Set units to millimeters
echo "Units set to millimeters."

send_gcode "G90"    # Set to absolute positioning
echo "Absolute positioning set."

# Wait to ensure the plotter is ready
sleep 2

echo "Starting drawing process..."
# Send G-code commands to draw a square
send_gcode "G1 X20 Y20 F1500"  # Move to the starting point (0,0)
# echo "Moved to starting point (0,0)."

send_gcode "G1 X80 Y0 F1000"  # Draw the first side (10mm along X-axis)
echo "Drawing first side."

send_gcode "G1 X80 Y80 F1000"  # Draw the second side (10mm along Y-axis)
echo "Drawing second side."

send_gcode "G1 X0 Y80 F1000"  # Draw the third side (back to Y=10, X=0)
echo "Drawing third side."

send_gcode "G1 X20 Y20 F1000"   # Draw the fourth side (back to origin)
echo "Drawing fourth side."

# send_gcode "G1 Z5 F500"    # Raise the pen after drawing
echo "Pen raised after drawing."

echo "Square drawn on the plotter."
