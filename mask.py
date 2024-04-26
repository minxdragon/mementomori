import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

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

