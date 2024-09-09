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
 
def main():

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
    #imshow(image)
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
        #imshow(mask)
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
    #imshow(mask)
    plt.show()

    # # Apply Sobel edge detection
    print("Applying Sobel edge detection...")
    sobelx = cv2.Sobel(binary_mask, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(binary_mask, cv2.CV_64F, 0, 1, ksize=5)
       # Combine the two results
    edges = cv2.magnitude(sobelx, sobely)

    # print("Applying Canny edge detection...")
    # edges = cv2.Canny(binary_mask, 100, 200)

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
    import xml.etree.ElementTree as ET

def crop_svg(svg_file, output_file):
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Initialize bounding box variables
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    # Loop through all elements to calculate the bounding box
    for elem in root.findall('.//svg:path', ns):  # You may need to adjust for different element types
        d = elem.get('d')
        if d:
            # Extract the coordinates from the 'd' attribute of the path
            path_commands = d.split(' ')
            for command in path_commands:
                try:
                    # Try to convert command to a float, which represents a coordinate
                    x_or_y = float(command)
                    if 'M' in path_commands or 'L' in path_commands:  # If it's a move or line command
                        min_x = min(min_x, x_or_y)
                        max_x = max(max_x, x_or_y)
                        # Assume next one is y
                        min_y = min(min_y, x_or_y)
                        max_y = max(max_y, x_or_y)
                except ValueError:
                    pass

    # Calculate width and height of the cropped content
    width = max_x - min_x
    height = max_y - min_y

    # Update the viewBox attribute to crop the SVG
    root.set('viewBox', f"{min_x} {min_y} {width} {height}")
    root.set('width', str(width))
    root.set('height', str(height))

    # Write the cropped SVG to a new file
    tree.write(output_file)
    print(f"Cropped SVG saved as {output_file}")

# Example usage


    # print("initializing plotter code")
    # svg_file = "output.svg"  # Replace with the actual SVG file name
    # gcode_file = "output.gcode"  # Output G-code file name

    # # # Convert SVG to G-code using Inkscape's command-line interface
    # # inkscape_command = f"inkscape {svg_file} --export-filename={gcode_file} --export-plain-svg --verb=Extensions.Gcodetools.Plot"
    # # subprocess.run(inkscape_command, shell=True)

    # # Send G-code to the plotter via serial connection
    # send_gcode_to_plotter(gcode_file)

if __name__ == "__main__":
    main()