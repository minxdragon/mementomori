import mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
import time

# every 30 minutes run mask.py
while True:
    print("Running mask.py")
    mask.main()
    time.sleep(1800)
    print("30 minutes have passed, running mask.py again")

    #get binary.png
