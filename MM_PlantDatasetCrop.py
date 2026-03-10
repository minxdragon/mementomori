from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- CONFIG ----------
INPUT_DIR = Path("uncropped").expanduser()
OUTPUT_DIR = Path("nature").expanduser()
DEBUG_DIR = Path("debug").expanduser()

TARGET_W = 1024
TARGET_H = 1024
PLANT_PADDING = 0.01

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.25

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

EXCLUDE_CLASSES = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "car", "bus", "truck",
    "motorcycle", "bicycle"
}

# Broad plant colour ranges
LOWER_GREEN = np.array([10, 15, 15], dtype=np.uint8)
UPPER_GREEN = np.array([105, 255, 255], dtype=np.uint8)

LOWER_BROWN = np.array([0, 10, 10], dtype=np.uint8)
UPPER_BROWN = np.array([40, 255, 255], dtype=np.uint8)

# Sky and water suppression
LOWER_BLUE = np.array([85, 20, 40], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)

# Optional neutral suppression, kept mild
LOWER_NEUTRAL = np.array([0, 0, 20], dtype=np.uint8)
UPPER_NEUTRAL = np.array([180, 40, 220], dtype=np.uint8)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(YOLO_MODEL)
class_names = model.names

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

def make_center_weight(h, w):
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dy = (y - cy) / (h / 2.0)
    dx = (x - cx) / (w / 2.0)
    dist = np.sqrt(dx * dx + dy * dy)
    weight = 1.0 - np.clip(dist, 0, 1)
    return (weight * 255).astype(np.uint8)

def build_exclusion_mask(img_bgr):
    h, w = img_bgr.shape[:2]
    exclusion = np.zeros((h, w), dtype=np.uint8)

    results = model.predict(source=img_bgr, conf=YOLO_CONF, verbose=False)
    r = results[0]

    found = []
    if r.boxes is None or len(r.boxes) == 0:
        return exclusion, found

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        cls_name = class_names[int(cls_id)]
        if cls_name not in EXCLUDE_CLASSES:
            continue

        x1, y1, x2, y2 = box.tolist()
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)

        x1 = clamp(x1 - pad_x, 0, w)
        y1 = clamp(y1 - pad_y, 0, h)
        x2 = clamp(x2 + pad_x, 0, w)
        y2 = clamp(y2 + pad_y, 0, h)

        cv2.rectangle(exclusion, (x1, y1), (x2, y2), 255, thickness=-1)
        found.append((cls_name, float(conf), x1, y1, x2, y2))

    return exclusion, found

def build_plant_mask(img_bgr, exclusion_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    brown_mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    neutral_mask = cv2.inRange(hsv, LOWER_NEUTRAL, UPPER_NEUTRAL)

    edges = cv2.Canny(gray, 60, 140)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    color_mask = cv2.bitwise_or(green_mask, brown_mask)
    color_support = cv2.dilate(color_mask, edge_kernel, iterations=2)
    structure_mask = cv2.bitwise_and(edges, color_support)

    plant_mask = cv2.bitwise_or(color_mask, structure_mask)

    plant_mask[exclusion_mask > 0] = 0
    plant_mask[blue_mask > 0] = 0

    neutral_penalty = cv2.bitwise_and(neutral_mask, cv2.bitwise_not(color_support))
    plant_mask[neutral_penalty > 0] = 0

    bottom_start = int(h * 0.62)
    bottom_green = green_mask[bottom_start:, :]
    if np.count_nonzero(bottom_green) > 0.15 * bottom_green.size:
        plant_mask[bottom_start:, :] = (plant_mask[bottom_start:, :].astype(np.float32) * 0.08).astype(np.uint8)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_big)
    plant_mask = cv2.dilate(plant_mask, kernel_small, iterations=2)

    # real centre weighting
    center_weight = make_center_weight(h, w).astype(np.float32) / 255.0
    weighted_mask = (plant_mask.astype(np.float32) * center_weight).astype(np.uint8)

    return weighted_mask, plant_mask, green_mask, brown_mask, blue_mask

def best_component_bbox(weighted_mask, binary_mask, img_h, img_w, min_area=800):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None

    best = None
    best_score = -1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        component_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        component_weights = weighted_mask[y:y+h, x:x+w]

        score = float((component_weights * component_mask).sum())

        # small penalty for huge boxes that swallow the frame
        box_area = w * h
        fill_ratio = area / max(box_area, 1)
        score *= (0.5 + fill_ratio)

        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)

    return best

def process_image(img_path: Path):
    rel = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    dbg_path = DEBUG_DIR / rel

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Unreadable: {img_path}")
        return

    h, w = img.shape[:2]
    exclusion_mask, excluded = build_exclusion_mask(img)
    weighted_mask, plant_mask, green_mask, brown_mask, blue_mask = build_plant_mask(img, exclusion_mask)
    # Only keep the strongest weighted regions for cropping
    crop_mask = np.zeros_like(weighted_mask, dtype=np.uint8)

    nz = weighted_mask[weighted_mask > 0]
    if nz.size > 0:
        thresh = np.percentile(nz, 75)   # try 75 first, then 80 or 85 if still loose
        crop_mask[weighted_mask >= thresh] = 255

    # connect nearby strong regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, kernel)
    crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)
    bbox = best_component_bbox(weighted_mask, crop_mask, h, w, min_area=500)
    coverage = np.count_nonzero(plant_mask) / (h * w)
    print(f"{img_path.name}: mask coverage {coverage:.4f}")

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, PLANT_PADDING)
        x1, y1, x2, y2 = fit_aspect(x1, y1, x2, y2, w, h, TARGET_W, TARGET_H)
        crop = img[y1:y2, x1:x2]
        mode = "plant_crop"
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
    else:
        crop = center_crop(img, TARGET_W, TARGET_H)
        mode = "center_fallback"

    crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    ensure_parent(out_path)
    ensure_parent(dbg_path)

    cv2.imwrite(str(out_path), crop)

    # Debug preview
    mask_vis = np.zeros_like(debug)
    mask_vis[:, :, 1] = crop_mask 
    mask_vis[:, :, 2] = exclusion_mask
    mask_vis[:, :, 0] = blue_mask

    preview = cv2.addWeighted(debug, 0.72, mask_vis, 0.35, 0)

    for cls_name, conf, x1, y1, x2, y2 in excluded:
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            preview,
            f"{cls_name} {conf:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        preview,
        f"{mode} cov={coverage:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(str(dbg_path), preview)
    print(f"Saved: {out_path}")

def main():
    print("Scanning:", INPUT_DIR)
    print("Exists?", INPUT_DIR.exists())
    print("Is dir?", INPUT_DIR.is_dir())

    all_paths = list(INPUT_DIR.rglob("*"))
    files = [p for p in all_paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    print(f"Found {len(files)} images")
    for i, img_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {img_path.name}")
        process_image(img_path)

if __name__ == "__main__":
    main()