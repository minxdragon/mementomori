import cv2
import replicate
import webbrowser
from urllib.request import urlopen, Request
import numpy as np
import re
import time
#import cairosvg
import xml.etree.ElementTree as ET
import subprocess

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
            words_to_replace = ["color", "colorful", "rainbow", "purple", "green", "blue", "brown", "red", "yellow", "orange", "pink", "beige", "turquoise", "teal", "navy", "maroon", "fuchsia", "gold", "silver", "bronze", "copper", "rainbow", "multicolor", "multicolored", "multicolour",]

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
        prompt = clip_interrogator(init_image)

        # Generate the image using the prompt and initial image
        model_name = "stability-ai/sdxl:8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f"
        output = replicate.run(
            model_name,
            input={"prompt": prompt+", coloring page, line art, plotter art", "image": open(init_image, "rb")}
        )

        # Assuming 'output' is a list containing the URL(s)
        for idx, url in enumerate(output):
            # Generate a filename based on the index
            filename = f'dream_{idx}.jpg'
            
            # Download the image from the URL and save it with the generated filename
            request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            req = urlopen(request_site)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # Load it as it is
            img = np.array(img)
            cv2.imwrite(filename, img)
            
            # Load and display the generated image using OpenCV
            generated_image = cv2.imread(filename)
            cv2.imshow("Generated Image", generated_image)
            print("Generated image saved as " + filename)
            # Convert the OpenCV image to SVG using autotrace
            print("Converting to SVG...")
            svg_filename = f'dream_{idx}.svg'
            # Run autotrace without the -output-file argument
            subprocess.run(['autotrace', filename,])

            cv2.waitKey(0)  # Wait for a key press before closing the window


            cv2.waitKey(0)  # Wait for a key press before closing the window

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    # Press Esc to exit
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()