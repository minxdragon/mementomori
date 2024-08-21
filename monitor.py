import os
from PIL import Image
import time
import subprocess

# Define the folder containing the PNGs
png_folder = "./pngs"

# Track processed images
processed_images = set()

# Create a blank base image (assuming 1920x1080, adjust as needed)
base_image = Image.new("RGBA", (1920, 1080), (255, 255, 255, 0))

def update_display():
    global processed_images, base_image
    
    # Get the list of PNG files
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])

    # Filter out the images that have already been processed
    new_files = [f for f in png_files if f not in processed_images]

    if not new_files:
        return  # No new files, so no need to update

    # Add each new PNG on top of the base image
    for png_file in new_files:
        new_layer = Image.open(os.path.join(png_folder, png_file))
        base_image.paste(new_layer, (0, 0), new_layer)
        processed_images.add(png_file)

    # Save the image temporarily to open it in full screen
    base_image.save("/tmp/fullscreen_image.png")

    # Close any previous instances of Preview
    subprocess.run(["pkill", "Preview"])

    # Open the image with Preview in full-screen mode
    subprocess.run(["open", "-a", "Preview", "/tmp/fullscreen_image.png"])

    # Add a delay to ensure Preview opens before triggering full screen
    time.sleep(1)

    # Trigger full-screen mode using AppleScript
    subprocess.run([
        "osascript", "-e",
        'tell application "Preview" to activate',
        "-e", 'tell application "System Events" to keystroke "f" using {control down, command down}'
    ])

# Run the display update loop
while True:
    update_display()
    time.sleep(5)  # Adjust the delay as needed
