import mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
import time
import glob
import os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

# Create a figure before the loop
plt.figure()

# Start the loop
while True:
    print("Running mask.py")
    mask.main()

    # Get a list of all the binary image files
    files = glob.glob('*.png')  # replace with your actual path and file pattern

    # Find the latest file
    latest_file = max(files, key=os.path.getctime)

    # Open the latest image
    latest_image = Image.open(latest_file)

    # Convert the image to a numpy array
    latest_image = np.array(latest_image)

    # Display the latest image on top of the previous images
    plt.imshow(latest_image, cmap='gray', alpha=0.5)

    # Draw the plot
    plt.draw()

    # Pause for a short period to allow the plot to update
    plt.pause(0.1)

    # Sleep 2 minutes
    time.sleep(120)
    print("2 minutes have passed, running mask.py again")