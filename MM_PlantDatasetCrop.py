from pathlib import Path
import cv2
import numpy as np

INPUT_DIR = Path("uncropped").expanduser()
OUTPUT_DIR = Path("nature").expanduser()
DEBUG_DIR = Path("debug").expanduser()

TARGET_W = 1024
TARGET_H = 1024
PADDING = 0.03

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

def process_image(img_path: Path):
    rel = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel
    dbg_path = DEBUG_DIR / rel

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Unreadable: {img_path}")
        return

    h, w = img.shape[:2]
    mask = leaf_mask(img)
    bbox = biggest_blob_bbox(mask, min_area=2000)

    debug = img.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, PADDING)
        x1, y1, x2, y2 = fit_aspect(x1, y1, x2, y2, w, h, TARGET_W, TARGET_H)
        crop = img[y1:y2, x1:x2]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        mode = "leaf_crop"
    else:
        crop = center_crop(img, TARGET_W, TARGET_H)
        mode = "center_fallback"

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
    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"Found {len(files)} images")
    for i, img_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {img_path.name}")
        process_image(img_path)

if __name__ == "__main__":
    main()