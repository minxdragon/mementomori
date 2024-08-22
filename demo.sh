#!/bin/bash

# Source and destination directories
SRC_DIR="/Users/j_laptop/Development/mementomori"
DEST_DIR="/Users/j_laptop/Development/mementomori/pngs"

# Create the destination directory if it doesn't exist
# mkdir -p "$DEST_DIR"

# Copy PNG files one by one with a 20-second delay
for file in "$SRC_DIR"/*.png; do
    if [ -f "$file" ]; then
        cp "$file" "$DEST_DIR"
        echo "Copied $(basename "$file") to $DEST_DIR"
        sleep 5  # 20-second delay
    fi
done

echo "All PNG files have been copied."
