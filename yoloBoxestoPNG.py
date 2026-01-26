import torch
import cv2
import xml.etree.ElementTree as ET

# YOLOv5 inference
def run_yolo(frame):
    # Load YOLOv5 model (you can use yolov5s, yolov5m, yolov5l, etc.)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform inference on the captured frame
    results = model(frame)
    
    # Extract bounding boxes for detected people
    persons = results.xyxy[0].cpu().numpy()
    
    return persons

# Function to create and save the SVG with bounding boxes
def create_svg_with_boxes(width, height, persons, output_filename):
    # Create an SVG file structure
    svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", width=str(width), height=str(height), version="1.1")

    for person in persons:
        x1, y1, x2, y2, confidence, class_id = person
        if class_id == 0:  # 'person' class in YOLO
            # Create a rectangle element for each bounding box
            rect = ET.Element('rect', {
                'x': str(int(x1)),
                'y': str(int(y1)),
                'width': str(int(x2 - x1)),
                'height': str(int(y2 - y1)),
                'stroke': 'black',
                'fill': 'none',
                'stroke-width': '2'
            })
            svg.append(rect)

    # Write the SVG to a file
    tree = ET.ElementTree(svg)
    tree.write(output_filename)
    print(f"SVG file saved as '{output_filename}'")

def main():
    # Capture an image from the webcam
    print("Running webcam...")
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        cap.release()
        return

    # Get the frame's dimensions
    height, width, _ = frame.shape

    # Run YOLO detection
    print("Running YOLO detection...")
    persons = run_yolo(frame)

    # create a unique filename for the SVG output using a random number and letter combination

    ufn = str(torch.randint(1000, (1,)).item()) + chr(torch.randint(97, 123, (1,)).item())

    # Create and save the png with bounding boxes
    output_filename = '/Users/j_laptop/mementomori/svgs/'+ufn+'.svg'
    create_svg_with_boxes(width, height, persons, output_filename)

    #convert to png
    print("Converting SVG to PNG...")
    import cairosvg
    cairosvg.svg2png(url=output_filename, write_to='/Users/j_laptop/mementomori/pngs/'+ufn+'.png')

    # Release webcam
    cap.release()

if __name__ == "__main__":
    main()
