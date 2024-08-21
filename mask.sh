#!/bin/bash

# Define file paths
SVG_DIR="/Users/j_laptop/Development/mementomori"
SVG_FILE="output.svg"
GCODE_FILE="output.gcode"
PYTHON_SCRIPT="mask.py"
#!/bin/bash

# Step 1: Run the Python script to generate the SVG
echo "Running Python script to generate SVG..."
python3 $PYTHON_SCRIPT

# # Step 2: Monitor for the creation of the SVG file
# echo "Waiting for SVG file to be created..."
# while [ ! -f "$SVG_DIR/$SVG_FILE" ]; do
#   sleep 1
# done

# Paths
INPUT_PNG="binary.png"
TEMP_SVG="temp_output.svg"
OUTPUT_GCODE="output.gcode"

# Convert PNG to SVG using Inkscape CLI
echo "Converting PNG to SVG..."
inkscape $INPUT_PNG --export-type=svg --export-filename=$TEMP_SVG

# Convert SVG to G-code using svg2gcode
echo "Converting SVG to G-code..."
svg2gcode $TEMP_SVG $OUTPUT_GCODE --cuttingspeed 1000 --cuttingpower 850 --selfcenter --showimage

echo "G-code file created: $OUTPUT_GCODE"


# Step 4: Send the G-code to the plotter
echo "Sending G-code to the plotter..."
# You can use Python or bash to send the G-code to the plotter
# Example using Python:
python3 <<EOF
import serial
import time

ser = serial.Serial('/dev/tty.usbserial-10', 115200)  # Replace with your serial port and baud rate

with open("$SVG_DIR/$GCODE_FILE", 'r') as f:
    for line in f:
        ser.write(line.encode() + b'\n')
        time.sleep(0.1)  # Small delay to ensure commands are processed

ser.close()
EOF

echo "G-code sent to plotter."

# Loop back to the beginning if you want this to be continuous
# Uncomment the next line if you want the script to loop
# exec "$0"
