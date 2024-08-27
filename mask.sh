#!/bin/bash

# Define file paths
SVG_DIR="/Users/j_laptop/Development/mementomori"
SVG_FILE="output.svg"
GCODE_FILE="output.gcode"
PYTHON_SCRIPT="mask.py"

# Step 1: Run the Python script to generate the SVG
echo "Running Python script to generate SVG..."
python3 $PYTHON_SCRIPT

# # Paths
# INPUT_PNG="binary.png"
# TEMP_SVG="temp_output.svg"
# OUTPUT_GCODE="output.gcode"

# # Convert PNG to SVG using Inkscape CLI
# echo "Converting PNG to SVG..."
# inkscape $INPUT_PNG --export-type=svg --export-filename=$TEMP_SVG

# # Check if the SVG file was created successfully
# if [ ! -f "$TEMP_SVG" ]; then
#   echo "Error: SVG file was not created."
#   exit 1
# fi

# # Convert SVG to G-code using svg2gcode
# echo "Converting SVG to G-code..."
# output=$(svg2gcode $TEMP_SVG $OUTPUT_GCODE --cuttingspeed 1000 --cuttingpower 850 --selfcenter --showimage 2>&1)
# echo "$output"

# # Check if the G-code file was created successfully
# if [ ! -f "$OUTPUT_GCODE" ]; then
#   echo "Error: G-code file was not created."
#   exit 1
# fi

#convert SVG to GCode using vpype
echo "Converting SVG to G-code..."
#vpype read $TEMP_SVG gwrite --profile gcode $OUTPUT_GCODE
vpype -v read /Users/j_laptop/Development/mementomori/output.svg   gwrite --profile gcode output.gcode
echo "G-code file created: $OUTPUT_GCODE"

# Step 4: Send the G-code to the plotter
echo "Sending G-code to the plotter..."
# You can use Python or bash to send the G-code to the plotter
# Example using Python:
python3 grbltest.py $GCODE_FILE

echo "G-code sent to plotter."

# Loop back to the beginning if you want this to be continuous
# Uncomment the next line if you want the script to loop
# exec "$0"
