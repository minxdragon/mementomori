import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply adaptive thresholding
binary_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply morphological operations
kernel = np.ones((3, 3), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Apply Canny edge detection
print("Applying Canny edge detection...")
edges = cv2.Canny(binary_mask, 100, 200)

# Normalize the pixel values to the range [0, 255]
edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Save the edge image
print("Saving edge image as 'mask_edges.jpg'")
cv2.imwrite('mask_edges.jpg', edges)

# Display the results
plt.subplot(121), plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])
plt.show()