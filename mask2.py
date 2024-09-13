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
import init
import xml.etree.ElementTree as ET
from svgoutline import svg_to_outlines
from PIL import Image
import matplotlib.pyplot as plt
from svgpathtools import svg2paths, wsvg, Path, Line
import uuid

def main():
    print("Loading model...")
    model = create_model("Unet_2020-07-20")
    model.eval()

    # Capture an image from the webcam
    print("Running webcam...")
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite('webcam.jpg', frame)
        print("Image captured and saved as 'webcam.jpg'")
    else:
        print("Failed to capture frame")
        cap.release()
        return

    cap.release()

    # Predict segmentation mask
    image = load_rgb("webcam.jpg")
    print("Predicting mask...")
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)

    # Invert the mask at the time of saving
    inverted_mask = cv2.bitwise_not(mask * 255)  # Invert to ensure inside is white, outside is black
    
    # Save the inverted mask as an image
    cv2.imwrite('mask_inverted.jpg', inverted_mask)  # Now the mask is inverted before saving
    print("Inverted mask saved as 'mask_inverted.jpg'")

    # Use the inverted mask for further processing (e.g., filling)
    print("Filling contours inside the inverted mask...")
    fill_mask = np.zeros_like(inverted_mask)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fill_mask, contours, -1, 255, thickness=-1)  # Fill inside the contours
    
    # Save the filled mask
    fill_file = 'fill.png'
    cv2.imwrite(fill_file, fill_mask)
    print(f"Filled mask saved as '{fill_file}'")

    # Apply 30% opacity to the filled image
    img_fill = Image.open(fill_file).convert("RGBA")
    fill_data = img_fill.getdata()
    new_fill_data = [(r, g, b, int(0.3 * a)) for (r, g, b, a) in fill_data]
    img_fill.putdata(new_fill_data)
    img_fill.save(fill_file)
    print(f"Filled image with 30% opacity saved as '{fill_file}'")

    # Assuming you generate an outline here (e.g., Sobel, Canny) and it's transparent outside.
    print("Generating outline...")
    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert edges to black outlines on a transparent background
    print("Creating black outlines on transparent background...")
    outline_image = np.zeros((edges.shape[0], edges.shape[1], 4), dtype=np.uint8)  # 4 channels (RGBA)
    outline_image[..., 0] = 0  # Set red channel to 0
    outline_image[..., 1] = 0  # Set green channel to 0
    outline_image[..., 2] = 0  # Set blue channel to 0 (black color)
    outline_image[..., 3] = edges  # Set alpha channel to edge intensity

    outline_file = 'outline.png'
    cv2.imwrite(outline_file, outline_image)
    print(f"Black outline on transparent background saved as '{outline_file}'")

    # Combine the outline and fill images without inverting anything
    img_outline = Image.open(outline_file).convert("RGBA")
    combined = Image.alpha_composite(img_outline, img_fill)

    # Save the final combined image
    # Create a unique filename for the combined image
    filename = uuid.uuid4().hex + '.png'
    combined_file = 'pngs/' + filename
    combined.save(combined_file)
    #create unique filename
    print(f"Combined image saved as '/pngs/{combined_file}'")

if __name__ == "__main__":
    main()
