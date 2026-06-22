from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def crop_to_leaf_density(image_path, model_path='yolo11x_leaf.pt', conf_thresh=0.15, 
                         grid_size=10, padding_ratio=0.1):
    """
    Crop image to the region with highest leaf detection density.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO leaf detection model
        conf_thresh: Confidence threshold for detections
        grid_size: Number of grid cells per dimension for density calculation
        padding_ratio: Padding around the crop region (as fraction of crop size)
    
    Returns:
        cropped_image: The cropped region with highest leaf density
        density_info: Dict with metadata about the crop
    """
    # Load model and run inference
    model = YOLO(model_path)
    results = model.predict(image_path, task="detect", save=False, conf=conf_thresh)
    
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print("No leaves detected")
        return image_rgb, {"error": "No detections"}
    
    # Extract bounding boxes from detections
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    
    # Create density grid
    grid_h = h / grid_size
    grid_w = w / grid_size
    density_grid = np.zeros((grid_size, grid_size))
    
    # Count leaf detections in each grid cell
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2 / grid_w)
        cy = int((y1 + y2) / 2 / grid_h)
        cx = min(max(cx, 0), grid_size - 1)
        cy = min(max(cy, 0), grid_size - 1)
        density_grid[cy, cx] += 1
    
    # Find grid cell with highest density
    max_density_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    max_density_cy, max_density_cx = max_density_idx
    max_density = density_grid[max_density_cy, max_density_cx]
    
    print(f"Max leaf density: {max_density} leaves at grid ({max_density_cx}, {max_density_cy})")
    
    # Get all boxes in the highest density region and nearby regions for expansion
    region_boxes = []
    search_radius = 2  # Look in adjacent cells too
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            gy = max_density_cy + dy
            gx = max_density_cx + dx
            if 0 <= gy < grid_size and 0 <= gx < grid_size:
                if density_grid[gy, gx] > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2 / grid_w)
                        cy = int((y1 + y2) / 2 / grid_h)
                        if cx == gx and cy == gy:
                            region_boxes.append(box)
    
    if not region_boxes:
        print("No boxes in high-density region, using all detections")
        region_boxes = boxes
    
    # Calculate bounding box of all leaves in high-density region
    region_boxes = np.array(region_boxes)
    crop_x1 = int(np.min(region_boxes[:, 0]))
    crop_y1 = int(np.min(region_boxes[:, 1]))
    crop_x2 = int(np.max(region_boxes[:, 2]))
    crop_y2 = int(np.max(region_boxes[:, 3]))
    
    # Add padding
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    pad_x = int(crop_w * padding_ratio)
    pad_y = int(crop_h * padding_ratio)
    
    crop_x1 = max(0, crop_x1 - pad_x)
    crop_y1 = max(0, crop_y1 - pad_y)
    crop_x2 = min(w, crop_x2 + pad_x)
    crop_y2 = min(h, crop_y2 + pad_y)
    
    cropped = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    
    info = {
        "original_size": (w, h),
        "crop_region": (crop_x1, crop_y1, crop_x2, crop_y2),
        "crop_size": (crop_x2 - crop_x1, crop_y2 - crop_y1),
        "num_detections": len(boxes),
        "num_in_region": len(region_boxes),
        "max_density": max_density,
        "max_density_grid_cell": (max_density_cx, max_density_cy),
    }
    
    return cropped, info


def main():
    image_path = 'file/directory'  # Replace with actual path
    model_path = 'yolo11x_leaf.pt'  # Replace if different
    
    # Crop to leaf density
    cropped_image, info = crop_to_leaf_density(image_path, model_path, conf_thresh=0.15)
    
    print(f"\nCrop info:")
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    # Display results
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.imshow(cropped_image)
    axes.set_title(f"Cropped to Leaf Density Region\n"
                   f"Detections in region: {info['num_in_region']}/{info['num_detections']}")
    axes.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Optionally save the cropped image
    # output_path = Path(image_path).stem + "_leaf_crop.jpg"
    # cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    # print(f"Saved cropped image to {output_path}")


if __name__ == "__main__":
    main()
