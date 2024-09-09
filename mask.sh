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

## vpype shaded version between points
# vpype -v read /Users/j_laptop/Development/mementomori/output.svg linesimplify linemerge gwrite --profile gcode output.gcode 

# # Optimized command for shading inside shapes
# vpype -v \
#   read "$SVG_DIR/$SVG_FILE" \
#   linemerge --tolerance 0.1mm \
#   reloop \
#   linesimplify --tolerance 0.02mm \
#   hatch -p 0.2mm -a 45 \
#   gwrite --profile gcode "$SVG_DIR/$GCODE_FILE"


#vpype outline version
vpype \
  read output.svg \
  linesort \
  linemerge --tolerance 0.2mm \
  layout --fit-to-margins 2cm a3 \
  linesimplify --tolerance 0.2mm \
  linesort \
  reloop \
  linesimplify \
  translate 15cm 10cm \
  gwrite --profile gcode output.gcode

#   vpype \
#   read output.svg \
#   linemerge --tolerance 0.2mm \
#   linesort \
#   linesimplify --tolerance 0.2mm \
#   crop 10mm 10mm 200mm 287mm \  # Adjust the crop dimensions to fit your specific needs
#   #hatch --distance 2mm --angle 45 \  # Optional, for hatching inside closed paths
#   layout --fit-to-margins 2cm a4 \
#   gwrite --profile gcode output.gcode

echo "G-code file created: $OUTPUT_GCODE"

# Step 4: Send the G-code to the plotter
echo "Sending G-code to the plotter..."
# You can use Python or bash to send the G-code to the plotter
# Example using Python:
python3 grbltest.py $GCODE_FILE

echo "G-code sent to plotter."

# Loop back to the beginning if you want this to be continuous
# Uncomment the next line if you want the script to loop
exec "$0"
