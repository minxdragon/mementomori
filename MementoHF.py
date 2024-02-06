import cv2
import replicate
import webbrowser
from urllib.request import urlopen, Request
import numpy as np
import re
import time
#import cairosvg
import xml.etree.ElementTree as ET
from PIL import Image 
import potrace
import requests

# Open the camera
cap = cv2.VideoCapture(0)

# Start time for the countdown
start_time = time.time()

# SET THE TIME INTERVAL TO SAVE FRAMES (in seconds)
SAVE_INTERVAL = 10  # Change this to your desired interval

# Variable to track whether the frame has been saved
frame_saved = False

while True:
    # Read each frame from the webcam
    ret, img = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        continue

    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the webcam feed in black and white
    cv2.imshow("Webcam Feed (Black & White)", grayscale_img)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Check if it's time to save a frame
    if elapsed_time >= SAVE_INTERVAL and not frame_saved:
        # Save the frame
        frame_filename = f"frame_{int(time.time())}.jpg"  # Use a timestamp as the filename
        cv2.imwrite(frame_filename, grayscale_img)  # Save the grayscale image
        frame_saved = True  # Mark the frame as saved

    # Check if 10 seconds have passed since saving the frame
    if frame_saved and time.time() - start_time >= SAVE_INTERVAL + 10:
        # Proceed to the next stage here
        # Replace the following with your desired code
        print("Next stage - Image has been saved for 10 seconds")
        break

    # Continue without waiting for a keypress if the frame has been saved
    if frame_saved:
        # Generate and display a new image
        prompt = ""
        init_image = frame_filename  # Use the saved frame as the input image

        # Define the clip_interrogator function
        import re

        # Define the clip_interrogator function
        def clip_interrogator(init_image):
            # Replace with the correct model and version
            model_name = "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70"

            # Use the provided image as input
            output = replicate.run(
                model_name,
                input={"image": open(init_image, "rb")}
            )

            print("Original output:")
            print(output)

            # Define a list of words to replace
            # create a dictionary of colors to replace
            words_to_replace = ["photographic", "photograph", "television", "video", "painting", "photo", "selfie", "red", "yellow", "orange", "pink", "beige", "turquoise", "teal", "navy", "maroon", "fuchsia", "gold", "silver", "bronze", "copper", "rainbow", "multicolor", "multicolored", "multicolour",]

            # Remove specified words using regular expressions
            cleaned_prompt = output
            print("Cleaning output..." + cleaned_prompt)
            for word in words_to_replace:
                print(f"Replacing '{word}' with ''")
                cleaned_prompt = re.sub(r'\b' + re.escape(word) + r'\b', '', cleaned_prompt, flags=re.IGNORECASE)

            # Print the content of cleaned_prompt after word replacement
            print("Cleaned output:")
            print(cleaned_prompt.strip())
            return cleaned_prompt.strip()

        # Run the clip_interrogator function
        prompt = "an etching of " + clip_interrogator(init_image) + ", cross hatching, etching, coloring page, outline, line art, plotter art"
        negative_prompt = "photograph, photorealistic, detailed"

        # Generate the image using the prompt and initial image


        API_URL = "https://api-inference.huggingface.co/models/stablediffusionapi/vector-art"
        headers = {"Authorization": f"Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        image_bytes = query({
            "inputs": prompt,
        })
        # You can access the image with PIL.Image for example
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))

        # # Assuming 'output' is a list containing the URL(s)
        # for idx, url in enumerate(output):
        #     # Generate a filename based on the index
        #     filename = f'dream_{idx}.jpg'
            
        #     # Download the image from the URL and save it with the generated filename
        #     request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        #     req = urlopen(request_site)
        #     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        #     img = cv2.imdecode(arr, -1)  # Load it as it is
        #     img = np.array(img)
        #     cv2.imwrite(filename, img)
            
        #     # Load and display the generated image using OpenCV
        #     # Load and display the generated image using OpenCV
        #     generated_image = cv2.imread(filename)
        #     cv2.imshow("Generated Image", generated_image)
        #     print("Generated image saved as " + filename)

        #     # Convert the OpenCV image to SVG using Potrace with edge detection
        #     print("Converting to SVG with edge detection...")

        #     # Resize the image to a smaller resolution before converting to grayscale
        #     resized_image = cv2.resize(generated_image, (240, 120))

        #     # Convert the resized image to grayscale
        #     gray_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        #     # Apply Canny edge detection to the grayscale image
        #     edges = cv2.Canny(gray_resized_image, 25, 100)

        #     # Display the edge-detected image
        #     cv2.imshow("Edge-Detected Image", edges)
        #     cv2.waitKey(0)  # Wait for a key press before closing the window

        #     # Pass the edge-detected image to Potrace with path simplification
        #     trace = potrace.Bitmap(edges)
        #     trace.turds = True  # Enable turd removal
        #     trace.alphamax = 0.1  # Adjust the alpha-max parameter
        #     path = trace.trace()

        #     # Get the SVG string representation of the path
        #     svg_string = path.to_svg()

        #     # Save the SVG string to an SVG file
        #     with open(f'dream_{idx}.svg', 'w') as svg_file:
        #         svg_file.write(svg_string)

        #     # Close the OpenCV window for the edge-detected image
        #     cv2.destroyAllWindows()



        #     path = trace.trace()

        #     # Save the path to an SVG file
        #     with open(f'dream_{idx}.svg', 'w') as svg_file:
        #         svg_file.write('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="240">\n')
                
        #         for curve in path:
        #             svg_file.write('<path d="')
        #             for segment in curve:
        #                 if segment.is_corner:
        #                     svg_file.write(f'M {segment.c[0]} {segment.c[1]} ')
        #                     svg_file.write(f'L {segment.end_point[0]} {segment.end_point[1]} ')
        #                 else:
        #                     svg_file.write(f'C {segment.c1[0]} {segment.c1[1]} {segment.c2[0]} {segment.c2[1]} {segment.end_point[0]} {segment.end_point[1]} ')
                    
        #             svg_file.write('" />\n')
                
        #         svg_file.write('</svg>')



        cv2.waitKey(0)  # Wait for a key press before closing the window

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    # Press Esc to exit
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()