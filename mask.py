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

<<<<<<< Updated upstream

print("Loading model...")
model = create_model("Unet_2020-07-20")
model.eval()

#save webcam as image
print("Running webcam...")
cap = cv2.VideoCapture(0)
=======
# def send_gcode_to_plotter(gcode_file):
#     # Adjust the serial settings for your plotter
#     ser = serial.Serial('/dev/tty.usbserial-10', 115200)  # Replace with the actual serial port and baud rate

#     with open(gcode_file, 'r') as f:
#         for line in f:
#             ser.write(line.encode() + b'\n')
#             time.sleep(0.1)  # Small delay to ensure commands are processed

#     ser.close()
    
def main():
>>>>>>> Stashed changes

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

# Normalize the pixel values to the range [0, 255]
mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Save the mask as an image
cv2.imwrite('mask.jpg', mask)
print("Mask saved as 'mask.jpg'")

# Convert the mask to a binary image
mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Save the binary mask as an image
cv2.imwrite('binary_mask.jpg', binary_mask)
print("Binary mask saved as 'binary_mask.jpg'")
imshow(mask)
plt.show()

# Apply Sobel edge detection
print("Applying Sobel edge detection...")
sobelx = cv2.Sobel(binary_mask, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(binary_mask, cv2.CV_64F, 0, 1, ksize=5)

# Combine the two results
edges = cv2.magnitude(sobelx, sobely)

# Normalize the pixel values to the range [0, 255]
edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Save the edge image
print("Saving edge image as 'mask_edges.jpg'")
cv2.imwrite('mask_edges.jpg', edges)

# Convert the edges image to a binary image
_, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Define the new size
new_size = (binary.shape[1] // 2, binary.shape[0] // 2)

# Resize the binary image
binary = cv2.resize(binary, new_size)

# Save the resized binary image
print("Saving resized binary image as 'binary.png'")
cv2.imwrite('binary.png', binary)

<<<<<<< Updated upstream
# Convert the binary image to SVG
print("Converting binary image to SVG...")
svg_data = init.main()
=======
    # Resize the binary image
    binary = cv2.resize(binary, new_size)

    # Save the resized binary image
    print("Saving resized binary image as 'binary.png'")
    filename = f'binary.png'

    # Save the binary image
    cv2.imwrite(filename, binary)

    #convert the white pixels to transparent
    img = Image.open(filename)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    img.save(filename, "PNG")

    # Convert the binary image to SVG
    print("Converting binary image to SVG...")
    init.main()

    # print("initializing plotter code")
    # svg_file = "output.svg"  # Replace with the actual SVG file name
    # gcode_file = "output.gcode"  # Output G-code file name

    # # Convert SVG to G-code using Inkscape's command-line interface
    # inkscape_command = f"inkscape {svg_file} --export-filename={gcode_file} --export-plain-svg --verb=Extensions.Gcodetools.Plot"
    # subprocess.run(inkscape_command, shell=True)

    # # Send G-code to the plotter via serial connection
    # send_gcode_to_plotter(gcode_file)

if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
