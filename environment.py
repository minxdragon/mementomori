import cv2
import numpy as np

# Load the pose estimation model (you may need to adjust the path)
net = cv2.dnn.readNetFromTensorflow('pose_model.pb')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to perform pose estimation
def estimate_pose(frame):
    # Your pose estimation logic here...
    # This is a simplified example using OpenCV's DNN module
    # You may need to fine-tune it based on your model and requirements

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Forward pass to get the pose estimation result
    output = net.forward()

    # Process the output (you may need to adapt this based on your model)
    keypoints = []
    for i in range(output.shape[1]):
        confidence_map = output[0, i, :, :]
        _, confidence, _, max_loc = cv2.minMaxLoc(confidence_map)
        if confidence > 0.5:
            keypoints.append(max_loc)

    # Compute bounding box
    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)
    bounding_box = [(x_min, y_min), (x_max, y_max)]

    return keypoints, bounding_box

# Function to map bounding box coordinates to plotter coordinates
def map_to_plotter_coordinates(bounding_box, webcam_frame_size, plotter_drawing_size):
    # Calculate scaling factors for mapping
    x_scale = plotter_drawing_size[0] / webcam_frame_size[0]
    y_scale = plotter_drawing_size[1] / webcam_frame_size[1]

    # Map bounding box coordinates to plotter coordinates
    mapped_bounding_box = [
        (int(bounding_box[0][0] * x_scale), int(bounding_box[0][1] * y_scale)),
        (int(bounding_box[1][0] * x_scale), int(bounding_box[1][1] * y_scale))
    ]

    return mapped_bounding_box

# Function to visualize bounding box on the frame
def visualize_on_frame(frame, mapped_bounding_box):
    # Draw a rectangle representing the bounding box on the frame
    cv2.rectangle(frame, mapped_bounding_box[0], mapped_bounding_box[1], (0, 255, 0), 2)

# Initialize the plotter size (adjust with the actual plotter size)
plotter_width, plotter_height = 800, 600

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform pose estimation
    keypoints, bounding_box = estimate_pose(frame)

    # Map bounding box to plotter coordinates
    plotter_drawing_size = (plotter_width, plotter_height)
    webcam_frame_size = (frame.shape[1], frame.shape[0])
    mapped_bounding_box = map_to_plotter_coordinates(bounding_box, webcam_frame_size, plotter_drawing_size)

    # Visualize the bounding box on the frame
    visualize_on_frame(frame, mapped_bounding_box)

    # Display the frame with keypoints and bounding box for visualization
    cv2.imshow('Pose Estimation', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()