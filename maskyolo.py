import cv2
import torch
import xml.etree.ElementTree as ET


def create_svg_with_boxes(width, height, detections, output_filename):
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(width),
        height=str(height),
        version="1.1",
    )

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        if int(class_id) == 0:  # person
            rect = ET.Element(
                "rect",
                {
                    "x": str(int(x1)),
                    "y": str(int(y1)),
                    "width": str(int(x2 - x1)),
                    "height": str(int(y2 - y1)),
                    "stroke": "black",
                    "fill": "none",
                    "stroke-width": "2",
                },
            )
            svg.append(rect)

    ET.ElementTree(svg).write(output_filename)
    print(f"Saved {output_filename}")


def draw_boxes(frame_bgr, detections, min_conf=0.25):
    out = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        if int(class_id) != 0:
            continue
        if float(conf) < min_conf:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main():
    print("Loading YOLOv5 model...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.25  # model-side confidence threshold

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0).")

    print("Press q to quit. Press s to save an SVG for the current frame.")

    last_detections = None
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # YOLO expects RGB; OpenCV is BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb)
        detections = results.xyxy[0].cpu().numpy()

        boxed = draw_boxes(frame, detections, min_conf=0.25)

        cv2.imshow("camera", frame)
        cv2.imshow("camera + boxes", boxed)

        last_detections = detections
        last_frame = frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and last_frame is not None and last_detections is not None:
            h, w, _ = last_frame.shape
            create_svg_with_boxes(w, h, last_detections, "output.svg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
