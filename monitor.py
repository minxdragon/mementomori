import os
from PIL import Image
import time
import subprocess

# Set your screen resolution here
screen_width = 1920  # Replace with your actual screen width
screen_height = 1080  # Replace with your actual screen height

# Define the folder containing the PNGs
png_folder = "./pngs"

# Track processed images
processed_images = set()

# Create a blank base image that matches the screen resolution
base_image = Image.new("RGBA", (screen_width, screen_height), (255, 255, 255, 0))

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
        
        # Resize new layer to match screen size
        new_layer = new_layer.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
        
        # Paste the new layer on the base image
        base_image.paste(new_layer, (0, 0), new_layer if new_layer.mode == 'RGBA' else None)
        processed_images.add(png_file)

    # Save the image temporarily to open it in full screen
    base_image.save("/tmp/fullscreen_image.png")

    # Open the image with Preview in full-screen mode
    subprocess.run(["open", "/tmp/fullscreen_image.png"])

# Run the display update loop
while True:
    update_display()
    time.sleep(3)  # Adjust the delay as needed
