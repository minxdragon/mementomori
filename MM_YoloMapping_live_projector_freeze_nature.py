import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Optional
import random

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw


# -----------------------------
# Geometry helpers
# -----------------------------
@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h


def geom_key(b: Box):
    return (b.x1, b.y1, b.x2, b.y2)


def clamp_box(b: Box, W: int, H: int) -> Box:
    x1 = max(0, min(b.x1, W - 1))
    y1 = max(0, min(b.y1, H - 1))
    x2 = max(0, min(b.x2, W))
    y2 = max(0, min(b.y2, H))
    if x2 <= x1 or y2 <= y1:
        return Box(0, 0, 0, 0)
    return Box(x1, y1, x2, y2)


def quantize_box(b: Box, q: int) -> Box:
    if q <= 1:
        return b
    x1 = (b.x1 // q) * q
    y1 = (b.y1 // q) * q
    x2 = ((b.x2 + q - 1) // q) * q
    y2 = ((b.y2 + q - 1) // q) * q
    return Box(x1, y1, x2, y2)


def boxes_from_yolo(persons, W: int, H: int, conf_thresh: float = 0.25, q: int = 8) -> List[Box]:
    out: List[Box] = []
    for det in persons:
        x1, y1, x2, y2, conf, class_id = det
        if int(class_id) != 0:
            continue
        if float(conf) < conf_thresh:
            continue
        b = Box(int(x1), int(y1), int(x2), int(y2))
        b = clamp_box(b, W, H)
        b = quantize_box(b, q)
        if b.w < 60 or b.h < 60:
            continue
        out.append(b)
    return out


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = a.area + b.area - inter
    return float(inter) / float(ua) if ua > 0 else 0.0


def lerp_box(a: Box, b: Box, t: float) -> Box:
    t = float(max(0.0, min(1.0, t)))
    return Box(
        int(round(a.x1 + (b.x1 - a.x1) * t)),
        int(round(a.y1 + (b.y1 - a.y1) * t)),
        int(round(a.x2 + (b.x2 - a.x2) * t)),
        int(round(a.y2 + (b.y2 - a.y2) * t)),
    )


# -----------------------------
# Tracking
# -----------------------------
@dataclass
class Track:
    track_id: int
    bbox: Box
    created_at: float
    last_seen_at: float
    frozen: bool = False
    frozen_box: Optional[Box] = None


# -----------------------------
# Rendering helpers
# -----------------------------
def draw_live_tracks_rgba(live_layer: Image.Image, tracks: List[Track], now: float):
    draw = ImageDraw.Draw(live_layer)
    purple = (180, 0, 255, 255)  # neon-ish purple
    for t in tracks:
        if t.frozen:
            continue
        age = max(0.0, now - t.created_at)
        width = int(3 + 2 * (0.5 + 0.5 * np.sin(age * 8.0)))  # subtle pulse
        b = t.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=purple, width=width)


def load_nature_images(nature_dir: Path) -> List[Image.Image]:
    if not nature_dir.exists():
        return []
    imgs: List[Image.Image] = []
    for p in sorted(nature_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    return imgs


def stamp_frozen_nature(accum: Image.Image, b: Box, nature_imgs: List[Image.Image]):
    """
    Stamp a filled rectangle into the permanent layer:
    - Fill with a resized crop from a random nature image.
    - Add a green outline so it reads clearly.
    Fallback: solid green fill if no nature images available.
    """
    draw = ImageDraw.Draw(accum)

    if b.area <= 0:
        return

    if nature_imgs:
        src = random.choice(nature_imgs)

        # Choose a crop with roughly the same aspect ratio as the box.
        tw, th = max(1, b.w), max(1, b.h)
        tar_ar = tw / th

        sw, sh = src.size
        src_ar = sw / sh

        if src_ar > tar_ar:
            # crop width
            new_w = int(round(sh * tar_ar))
            x0 = random.randint(0, max(0, sw - new_w))
            crop = src.crop((x0, 0, x0 + new_w, sh))
        else:
            # crop height
            new_h = int(round(sw / tar_ar))
            y0 = random.randint(0, max(0, sh - new_h))
            crop = src.crop((0, y0, sw, y0 + new_h))

        patch = crop.resize((tw, th), resample=Image.BILINEAR).convert("RGBA")
        accum.alpha_composite(patch, (b.x1, b.y1))
    else:
        fill = (0, 255, 0, 255)
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], fill=fill)

    outline = (80, 255, 80, 255)
    draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=outline, width=2)


def pil_rgba_to_bgr(im: Image.Image) -> np.ndarray:
    arr = np.array(im, dtype=np.uint8)  # RGBA
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = Path(os.path.expanduser("~/mementomori"))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Put nature images here:
    # ~/mementomori/nature/
    nature_dir = base_dir / "nature"
    nature_imgs = load_nature_images(nature_dir)

    output_dir = base_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = base_dir / "latest.png"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    acc_frozen: Optional[Image.Image] = None

    tracks: List[Track] = []
    next_id = 1
    frozen_keys = set()

    # Tuning
    Q = 24                 # quantize to reduce jitter
    CONF = 0.25
    DETECT_EVERY = 2       # YOLO every N frames
    SMOOTH_T = 0.25        # 0..1
    LIVE_SECONDS = 4.0     # purple live duration
    MISS_SECONDS = 0.8     # tolerance for missed detections
    IOU_THRESH = 0.20

    # Display
    WIN = "mementomori_live"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Try to go fullscreen on the projector window.
    # If this annoys you while debugging, comment these two lines.
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        now = time.time()
        H, W = frame.shape[:2]

        if acc_frozen is None or acc_frozen.size != (W, H):
            acc_frozen = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            tracks.clear()
            next_id = 1
            frozen_keys.clear()

        detections: List[Box] = []
        if frame_idx % DETECT_EVERY == 0:
            results = model(frame)
            persons = results.xyxy[0].cpu().numpy()
            detections = boxes_from_yolo(persons, W, H, conf_thresh=CONF, q=Q)

        # Match detections to existing LIVE tracks (greedy IoU)
        unmatched_det = set(range(len(detections)))
        live_tracks_idx = [i for i, t in enumerate(tracks) if not t.frozen]
        used_tracks = set()

        for di, det in enumerate(detections):
            best_i = -1
            best_s = 0.0
            for ti in live_tracks_idx:
                if ti in used_tracks:
                    continue
                s = iou(det, tracks[ti].bbox)
                if s > best_s:
                    best_s = s
                    best_i = ti
            if best_i >= 0 and best_s >= IOU_THRESH:
                t = tracks[best_i]
                tracks[best_i].bbox = lerp_box(t.bbox, det, SMOOTH_T)
                tracks[best_i].last_seen_at = now
                used_tracks.add(best_i)
                unmatched_det.discard(di)

        # Create new tracks for unmatched detections
        for di in sorted(unmatched_det):
            det = detections[di]
            tracks.append(Track(track_id=next_id, bbox=det, created_at=now, last_seen_at=now))
            next_id += 1

        # Drop stale live tracks
        tracks = [
            t for t in tracks
            if t.frozen or (now - t.last_seen_at) <= MISS_SECONDS
        ]

        # Freeze tracks that have existed long enough
        froze_any = False
        for t in tracks:
            if t.frozen:
                continue
            if (now - t.created_at) >= LIVE_SECONDS:
                t.frozen = True
                qb = quantize_box(clamp_box(t.bbox, W, H), Q)
                t.frozen_box = qb
                k = geom_key(qb)
                if qb.area > 0 and k not in frozen_keys:
                    frozen_keys.add(k)
                    stamp_frozen_nature(acc_frozen, qb, nature_imgs)
                    froze_any = True

        # Compose output: frozen layer + live layer
        live_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_live_tracks_rgba(live_layer, tracks, now)

        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.alpha_composite(acc_frozen)
        out.alpha_composite(live_layer)

        # Live projector view
        bgr = pil_rgba_to_bgr(out)
        cv2.imshow(WIN, bgr)

        # Save only when something freezes (plus update latest.png)
        if froze_any:
            frame_path = output_dir / f"fixed_{saved_idx:06d}.png"
            out.save(frame_path)

            tmp_latest = latest_path.with_suffix(".tmp.png")
            out.save(tmp_latest)
            os.replace(tmp_latest, latest_path)

            saved_idx += 1

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
