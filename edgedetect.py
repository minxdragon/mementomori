import cv2
import numpy as np

# Read an image (replace 'your_image.jpg' with the actual image file)
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150)

# Display the original image and the edges side by side
cv2.imshow('Original Image', image)
cv2.imshow('Edge Detection', edges)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
