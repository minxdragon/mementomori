import mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
import time
import glob
import os
from PIL import Image

# every 30 minutes run mask.py
while True:
    print("Running mask.py")
    mask.main()
    time.sleep(1800)
    print("30 minutes have passed, running mask.py again")
    # Get a list of all the binary image files
    files = glob.glob('*.png')  # replace with your actual path and file pattern

    # Find the latest file
    latest_file = max(files, key=os.path.getctime)

    # Open the latest image
    latest_image = Image.open(latest_file)

    # Convert the image to a numpy array
    latest_image = np.array(latest_image)

    # Display the latest image
    plt.imshow(latest_image, cmap='gray')
    plt.show()

    
