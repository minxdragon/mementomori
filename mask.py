import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from PngToSvg import init


print("Loading model...")
model = create_model("Unet_2020-07-20")
model.eval()

#save webcam as image
print("Running webcam...")
cap = cv2.VideoCapture(0)

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Save the frame as an image
    cv2.imwrite('webcam.jpg', frame)
    print("Image captured and saved as 'webcam.jpg'")
else:
    print("Failed to capture frame")

cap.release()
image = load_rgb("webcam.jpg")
imshow(image)
print("Predicting mask...")
transform = albu.Compose([albu.Normalize(p=1)], p=1)
padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
with torch.no_grad():
  prediction = model(x)[0][0]
  mask = (prediction > 0).cpu().numpy().astype(np.uint8)
  mask = unpad(mask, pads)
  print("Mask predicted")
  imshow(mask)
  plt.show()

# save mask as image
cv2.imwrite('mask.jpg', mask)
print("Mask saved as 'mask.jpg'")

# Convert the mask to grayscale if it's not already
print("Converting mask to grayscale...")
if len(mask.shape) == 3:
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    gray_mask = mask

# Apply Sobel edge detection
print("Applying Sobel edge detection...")
sobelx = cv2.Sobel(gray_mask, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_mask, cv2.CV_64F, 0, 1, ksize=5)

# Combine the two results
edges = cv2.magnitude(sobelx, sobely)

# Save the edge image
print("Saving edge image as 'mask_edges.jpg'")
cv2.imwrite('mask_edges.jpg', edges)

# Convert the edges image to a binary image
_, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Save the binary image
cv2.imwrite('binary.png', binary)

# Convert the binary image to SVG
svg_data = init.maskmain('binary.png')

# Save the SVG data to a file
with open('mask_edges.svg', 'w') as f:
    f.write(svg_data)