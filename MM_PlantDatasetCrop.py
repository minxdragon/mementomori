from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

INPUT_DIR = Path("uncropped").expanduser()
OUTPUT_DIR = Path("nature").expanduser()
DEBUG_DIR = Path("debug").expanduser()

TARGET_W = 1024
TARGET_H = 1024
PADDING = 0.03
YOLO_MODEL_PATHS = [Path("yolo11x_leaf.pt"), Path("plants.pt"), Path("plant_yolov8.pt"), Path("yolov8n.pt")]
YOLO_CONF = 0.15
LEAF_DENSITY_GRID_SIZE = 8
LEAF_DENSITY_PADDING = 0.1
MIN_CROP_SIZE_RATIO = 0.4  # Minimum 40% of smallest image dimension

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def fit_aspect(x1, y1, x2, y2, img_w, img_h, target_w, target_h):
    target_ratio = target_w / target_h
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_ratio = bw / bh

    if box_ratio > target_ratio:
        new_w = bw
        new_h = int(round(new_w / target_ratio))
    else:
        new_h = bh
        new_w = int(round(new_h * target_ratio))

    x1 = int(round(cx - new_w / 2))
    x2 = int(round(cx + new_w / 2))
    y1 = int(round(cy - new_h / 2))
    y2 = int(round(cy + new_h / 2))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h

    return clamp(x1, 0, img_w), clamp(y1, 0, img_h), clamp(x2, 0, img_w), clamp(y2, 0, img_h)

def center_crop(img, target_w, target_h):
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x1 = (w - new_w) // 2
        return img[:, x1:x1 + new_w]
    else:
        new_h = int(round(w / target_ratio))
        y1 = (h - new_h) // 2
        return img[y1:y1 + new_h, :]

def expand_bbox(x1, y1, x2, y2, img_w, img_h, padding):
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * padding))
    pad_y = int(round(bh * padding))
    return (
        clamp(x1 - pad_x, 0, img_w),
        clamp(y1 - pad_y, 0, img_h),
        clamp(x2 + pad_x, 0, img_w),
        clamp(y2 + pad_y, 0, img_h),
    )


def load_yolo_model():
    if YOLO is None:
        return None

    for path in YOLO_MODEL_PATHS:
        if path.exists():
            try:
                return YOLO(str(path))
            except Exception as exc:
                print(f"Warning: failed to load YOLO model {path}: {exc}")
    return None


def detect_main_plant_bbox(img_bgr, model, target_class=None, conf_thresh=YOLO_CONF):
    try:
        results = model(img_bgr, conf=conf_thresh, iou=0.45, verbose=False)
    except Exception as exc:
        print(f"Warning: YOLO inference failed: {exc}")
        return None

    if not results or len(results) == 0:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    names = getattr(result, "names", {})

    valid_indices = list(range(len(xyxy)))
    if target_class is not None:
        valid_indices = [i for i, cls in enumerate(classes)
                         if str(names.get(cls, cls)).lower() == str(target_class).lower()]
        if not valid_indices:
            return None

    def score(i):
        w = xyxy[i, 2] - xyxy[i, 0]
        h = xyxy[i, 3] - xyxy[i, 1]
        return confs[i] * (w * h)

    best_idx = max(valid_indices, key=score)
    x1, y1, x2, y2 = xyxy[best_idx].astype(int)
    return x1, y1, x2, y2


def detect_leaf_density_bbox(img_bgr, model, grid_size=LEAF_DENSITY_GRID_SIZE, conf_thresh=YOLO_CONF):
    """Detect the region with highest leaf concentration and return its bounding box."""
    try:
        results = model(img_bgr, conf=conf_thresh, iou=0.45, verbose=False)
    except Exception as exc:
        return None

    if not results or len(results) == 0:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    h, w = img_bgr.shape[:2]

    # Create density grid
    grid_h = h / grid_size
    grid_w = w / grid_size
    density_grid = np.zeros((grid_size, grid_size))

    # Count detections in each grid cell
    for box in xyxy:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2 / grid_w)
        cy = int((y1 + y2) / 2 / grid_h)
        cx = min(max(cx, 0), grid_size - 1)
        cy = min(max(cy, 0), grid_size - 1)
        density_grid[cy, cx] += 1

    # Find highest density region
    max_density_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    max_density_cy, max_density_cx = max_density_idx
    max_density = density_grid[max_density_cy, max_density_cx]

    # Gather boxes in high-density region and nearby cells
    region_boxes = []
    
    # Expand search radius if few detections found
    if max_density <= 1:
        search_radius = 3
    elif max_density <= 3:
        search_radius = 2
    else:
        search_radius = 1
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            gy = max_density_cy + dy
            gx = max_density_cx + dx
            if 0 <= gy < grid_size and 0 <= gx < grid_size:
                if density_grid[gy, gx] > 0:
                    for box in xyxy:
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2 / grid_w)
                        cy = int((y1 + y2) / 2 / grid_h)
                        if cx == gx and cy == gy:
                            region_boxes.append(box)

    if not region_boxes:
        region_boxes = xyxy

    region_boxes = np.array(region_boxes)
    crop_x1 = int(np.min(region_boxes[:, 0]))
    crop_y1 = int(np.min(region_boxes[:, 1]))
    crop_x2 = int(np.max(region_boxes[:, 2]))
    crop_y2 = int(np.max(region_boxes[:, 3]))

    # Add padding - scale up padding for sparse detections
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # Increase padding multiplier if few detections
    num_detections = len(region_boxes)
    if num_detections == 1:
        padding_scale = LEAF_DENSITY_PADDING * 3.0  # Triple padding for single detection
    elif num_detections <= 3:
        padding_scale = LEAF_DENSITY_PADDING * 2.0  # Double padding for 2-3 detections
    else:
        padding_scale = LEAF_DENSITY_PADDING
    
    pad_x = max(int(crop_w * padding_scale), int(w * 0.05))  # Minimum 5% of image width
    pad_y = max(int(crop_h * padding_scale), int(h * 0.05))  # Minimum 5% of image height

    crop_x1 = max(0, crop_x1 - pad_x)
    crop_y1 = max(0, crop_y1 - pad_y)
    crop_x2 = min(w, crop_x2 + pad_x)
    crop_y2 = min(h, crop_y2 + pad_y)

    # Enforce minimum crop size to prevent over-zooming
    min_size = int(min(h, w) * MIN_CROP_SIZE_RATIO)
    crop_w_final = crop_x2 - crop_x1
    crop_h_final = crop_y2 - crop_y1
    
    if crop_w_final < min_size or crop_h_final < min_size:
        # Center the crop region and expand to minimum size
        cx = (crop_x1 + crop_x2) / 2.0
        cy = (crop_y1 + crop_y2) / 2.0
        
        new_w = max(crop_w_final, min_size)
        new_h = max(crop_h_final, min_size)
        
        crop_x1 = int(max(0, cx - new_w / 2))
        crop_x2 = int(min(w, cx + new_w / 2))
        crop_y1 = int(max(0, cy - new_h / 2))
        crop_y2 = int(min(h, cy + new_h / 2))
        
        # Adjust if hit image boundaries
        if crop_x2 - crop_x1 < new_w:
            if crop_x1 == 0:
                crop_x2 = min(w, crop_x1 + new_w)
            else:
                crop_x1 = max(0, crop_x2 - new_w)
        
        if crop_y2 - crop_y1 < new_h:
            if crop_y1 == 0:
                crop_y2 = min(h, crop_y1 + new_h)
            else:
                crop_y1 = max(0, crop_y2 - new_h)

    return crop_x1, crop_y1, crop_x2, crop_y2


def enhance_crop(crop_bgr):
    crop = crop_bgr if crop_bgr.dtype == np.uint8 else np.clip(crop_bgr, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = 1.08
    beta = 5
    crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    return crop


def leaf_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # green leaves
    lower_green = np.array([25, 35, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # keep only stronger greens
    nz = mask > 0
    if np.count_nonzero(nz) == 0:
        return mask

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # require moderate saturation and brightness
    strong = ((sat > 50) & (val > 40) & nz).astype(np.uint8) * 255

    # clean and connect leaves
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, kernel1)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, kernel2)
    strong = cv2.dilate(strong, kernel1, iterations=2)

    return strong

def biggest_blob_bbox(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    best_idx = None
    best_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_idx = i

    if best_idx is None:
        return None

    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]
    return x, y, x + w, y + h

def process_image(img_path: Path, yolo_model=None):
    rel = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    dbg_path = DEBUG_DIR / rel

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Unreadable: {img_path}")
        return

    h, w = img.shape[:2]
    debug = img.copy()
    mask = leaf_mask(img)

    bbox = None
    mode = None
    
    # Priority 1: Try leaf density detection (for leaf-specific models like yolo11x_leaf)
    if yolo_model is not None:
        leaf_bbox = detect_leaf_density_bbox(img, yolo_model)
        if leaf_bbox is not None:
            bbox = leaf_bbox
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 165, 0), 3)  # Orange
            mode = "leaf_density"
    
    # Priority 2: Try general plant detection
    if bbox is None and yolo_model is not None:
        plant_bbox = detect_main_plant_bbox(img, yolo_model)
        if plant_bbox is not None:
            bbox = plant_bbox
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 3)
            mode = "yolo_crop"

    # Priority 3: Color-based leaf mask detection
    if bbox is None:
        bbox = biggest_blob_bbox(mask, min_area=2000)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, PADDING)
            x1, y1, x2, y2 = fit_aspect(x1, y1, x2, y2, w, h, TARGET_W, TARGET_H)
            crop = img[y1:y2, x1:x2]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            mode = "leaf_mask"
    
    # Priority 4: Center fallback
    if bbox is None:
        crop = center_crop(img, TARGET_W, TARGET_H)
        mode = "center_fallback"
    else:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, PADDING)
        x1, y1, x2, y2 = fit_aspect(x1, y1, x2, y2, w, h, TARGET_W, TARGET_H)
        crop = img[y1:y2, x1:x2]

    crop = enhance_crop(crop)
    crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    ensure_parent(out_path)
    ensure_parent(dbg_path)

    cv2.imwrite(str(out_path), crop)

    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    preview = cv2.addWeighted(debug, 0.75, mask_vis, 0.25, 0)
    cv2.putText(preview, mode, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(dbg_path), preview)
    print(f"Saved: {out_path}")

def main():
    yolo_model = load_yolo_model()
    if yolo_model is not None:
        model_name = getattr(yolo_model, "model_path", None) or getattr(yolo_model, "path", None) or "YOLO model"
        print(f"Using YOLO model: {model_name}")
    else:
        if YOLO is None:
            print("YOLO not available: falling back to color-based plant crop")
        else:
            print("No YOLO model found: falling back to color-based plant crop")

    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"Found {len(files)} images")
    for i, img_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {img_path.name}")
        process_image(img_path, yolo_model=yolo_model)

if __name__ == "__main__":
    main()