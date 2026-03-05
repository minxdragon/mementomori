
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import random
import hashlib

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from collections import deque


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

def add_box_edges(Hedge, Vedge, b, Q):
    # b is a quantized Box in pixel coords
    gx0 = 1 + max(0, min(GW0, b.x1 // Q))
    gx1 = 1 + max(0, min(GW0, b.x2 // Q))
    gy0 = 1 + max(0, min(GH0, b.y1 // Q))
    gy1 = 1 + max(0, min(GH0, b.y2 // Q))
    if gx1 <= gx0 or gy1 <= gy0:
        return
    Hedge[gy0, gx0:gx1] = 1
    Hedge[gy1, gx0:gx1] = 1
    Vedge[gy0:gy1, gx0] = 1
    Vedge[gy0:gy1, gx1] = 1

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

def find_closed_rectangles_on_grid(Hedge, Vedge):
    GH = Vedge.shape[0]
    GW = Hedge.shape[1]
    Q = 24
    visited = np.zeros((GH, GW), dtype=np.uint8)
    rects = []

    def neighbors(y, x):
        # up
        if y > 0 and Hedge[y, x] == 0:
            yield (y - 1, x)
        # down
        if y < GH - 1 and Hedge[y + 1, x] == 0:
            yield (y + 1, x)
        # left
        if x > 0 and Vedge[y, x] == 0:
            yield (y, x - 1)
        # right
        if x < GW - 1 and Vedge[y, x + 1] == 0:
            yield (y, x + 1)

    for sy in range(GH):
        for sx in range(GW):
            if visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = 1

            cells = 0
            minx = maxx = sx
            miny = maxy = sy
            touches_border = (sx == 0 or sy == 0 or sx == GW - 1 or sy == GH - 1)

            while q:
                y, x = q.popleft()
                cells += 1
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y
                if x == 0 or y == 0 or x == GW - 1 or y == GH - 1:
                    touches_border = True

                for ny, nx in neighbors(y, x):
                    if not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))

            if touches_border:
                continue

            w = (maxx - minx + 1)
            h = (maxy - miny + 1)
            if cells != w * h:
                continue  # enclosed but not a rectangle

            # convert cell bbox to pixel bbox
            x1, y1 = minx * Q, miny * Q
            x2, y2 = (maxx + 1) * Q, (maxy + 1) * Q
            rects.append((x1, y1, x2, y2))

    return rects
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


def stamp_frozen_fill(acc_fill: Image.Image,
                     b: Box,
                     nature_imgs: List[Image.Image],
                     *,
                     fill_alpha: int = 255,
                     rng: Optional[random.Random] = None):
    """Fill a rectangle with a nature image crop (overwrite).
    Used when repainting all detected closed rectangles.
    """
    if b.w <= 0 or b.h <= 0:
        return

    W, H = acc_fill.size
    b = clamp_box(b, W, H)
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
    if x2 <= x1 or y2 <= y1:
        return

    tw, th = x2 - x1, y2 - y1

    if nature_imgs:
        if rng is None:
            rng = random
        src = rng.choice(nature_imgs)
        sw, sh = src.size
        if sw <= 1 or sh <= 1:
            return

        target_ar = tw / th
        src_ar = sw / sh

        if src_ar >= target_ar:
            new_w = max(1, int(sh * target_ar))
            x0 = rng.randint(0, max(0, sw - new_w))
            crop = src.crop((x0, 0, x0 + new_w, sh))
        else:
            new_h = max(1, int(sw / target_ar))
            y0 = rng.randint(0, max(0, sh - new_h))
            crop = src.crop((0, y0, sw, y0 + new_h))

        patch = crop.resize((tw, th), resample=Image.BILINEAR).convert("RGBA")
    else:
        patch = Image.new("RGBA", (tw, th), (0, 255, 0, 255))

    if fill_alpha < 255:
        arr = np.array(patch, dtype=np.uint8)
        a = arr[:, :, 3].astype(np.uint16)
        arr[:, :, 3] = (a * fill_alpha // 255).astype(np.uint8)
        patch = Image.fromarray(arr, mode="RGBA")

    acc_fill.paste(patch, (x1, y1))
def stamp_frozen_outline(acc_lines: Image.Image, b: Box):
    """Stamp a permanent green outline into the persistent lines layer."""
    draw = ImageDraw.Draw(acc_lines)
    green = (0, 255, 0, 255)
    

    # Double-stroke: thick + thin gives density and intersection legibility
    draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=green, width=6)
    draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=green, width=2)


def pil_rgba_to_bgr(im: Image.Image) -> np.ndarray:
    arr = np.array(im, dtype=np.uint8)  # RGBA
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)




def detect_closed_rectangles(acc_lines: Image.Image,
                            *,
                            min_w: int = 24,
                            min_h: int = 24,
                            dilate_px: int = 1,
                            wall_thresh: int = 200,
                            extent_thresh: float = 0.80,
                            wall_side_thresh: float = 0.65) -> List[Tuple[int, int, int, int]]:
    """Detect enclosed axis-aligned rectangular regions implied by the current line layer.

    Returns a list of rectangle bounds (x1,y1,x2,y2). Designed to be tolerant of thick/AA lines.
    """
    arr = np.array(acc_lines, dtype=np.uint8)  # RGBA
    alpha = arr[:, :, 3]

    wall = (alpha >= wall_thresh).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        wall = cv2.dilate(wall, np.ones((k, k), np.uint8), iterations=1)

    H, W = wall.shape[:2]

    wall[0, :] = 255
    wall[-1, :] = 255
    wall[:, 0] = 255
    wall[:, -1] = 255

    free = (wall == 0).astype(np.uint8)

    num, labels = cv2.connectedComponents(free, connectivity=4)
    if num <= 1:
        return []

    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])))

    def side_wall_ratio(x1, y1, x2, y2) -> float:
        top = wall[y1, x1:x2]
        bot = wall[y2 - 1, x1:x2]
        left = wall[y1:y2, x1]
        right = wall[y1:y2, x2 - 1]
        tot = (top.size + bot.size + left.size + right.size)
        if tot <= 0:
            return 0.0
        hit = (np.count_nonzero(top) + np.count_nonzero(bot) +
               np.count_nonzero(left) + np.count_nonzero(right))
        return hit / tot

    rects: List[Tuple[int, int, int, int]] = []

    for lab in range(1, num):
        if lab in border_labels:
            continue

        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        w, h = x2 - x1, y2 - y1
        if w < min_w or h < min_h:
            continue

        area = int(xs.size)
        rect_area = int(w * h)
        if rect_area <= 0:
            continue

        extent = area / rect_area
        if extent < extent_thresh:
            continue

        if side_wall_ratio(x1, y1, x2, y2) < wall_side_thresh:
            continue

        rects.append((x1, y1, x2, y2))

    return rects
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

    acc_fill: Optional[Image.Image] = None
    acc_lines: Optional[Image.Image] = None

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

        if acc_fill is None or acc_lines is None or acc_fill.size != (W, H) or acc_lines.size != (W, H):
            acc_fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            acc_lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))

            tracks.clear()
            next_id = 1
            frozen_keys.clear()
            frozen_boxes: List[Box] = []

            # --- grid geometry ---
            GW0, GH0 = W // Q, H // Q
            GW, GH = GW0 + 2, GH0 + 2

            Hedge = np.zeros((GH + 1, GW), dtype=np.uint8)
            Vedge = np.zeros((GH, GW + 1), dtype=np.uint8)

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
        frozen_boxes_this_frame: List[Box] = []
        
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
                    stamp_frozen_outline(acc_lines, qb)
                    add_box_edges(Hedge, Vedge, qb, Q)
                    if qb.area > 0 and k not in frozen_keys:
                        frozen_keys.add(k)
                        stamp_frozen_outline(acc_lines, qb)
                        add_box_edges(Hedge, Vedge, qb, Q)
                        frozen_boxes.append(qb)   # <-- add here
                        frozen_boxes_this_frame.append(qb)
                        froze_any = True

                    frozen_boxes_this_frame.append(qb)
                    froze_any = True

        if froze_any:
            rects = find_closed_rectangles_on_grid(Hedge, Vedge)
            print("grid closed rects:", len(rects))

            # repaint fills: big first, small on top
            rects = sorted(set(rects), key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
            acc_fill.paste((0, 0, 0, 0), (0, 0, W, H))

# fill frozen YOLO boxes first
        for b in frozen_boxes:
            key = f"box:{b.x1},{b.y1},{b.x2},{b.y2}".encode("utf-8")
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            rng = random.Random(seed)
            stamp_frozen_fill(acc_fill, b, nature_imgs, fill_alpha=255, rng=rng)

        # then fill detected closed cells on top (smaller ones win)
        for (x1, y1, x2, y2) in rects:
            key = f"cell:{x1},{y1},{x2},{y2}".encode("utf-8")
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            rng = random.Random(seed)
            stamp_frozen_fill(acc_fill, Box(x1, y1, x2, y2), nature_imgs, fill_alpha=255, rng=rng)
            # Try to repaint fills for every closed rectangle so each cell gets its own image.
            rects = detect_closed_rectangles(
                acc_lines,
                min_w=Q,
                min_h=Q,
                dilate_px=0,
                wall_thresh=40,
                extent_thresh=0.55,
                wall_side_thresh=0.45,
            )

            # Only clear and repaint if we actually found any rectangles.
            if rects:
                # Paint big first, then small on top, so small cells keep their own image.
                rects = sorted(set(rects), key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
                acc_fill.paste((0, 0, 0, 0), (0, 0, W, H))
                for (x1, y1, x2, y2) in rects:
                    key = f"{x1},{y1},{x2},{y2}".encode("utf-8")
                    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
                    rng = random.Random(seed)
                    stamp_frozen_fill(acc_fill, Box(x1, y1, x2, y2), nature_imgs, fill_alpha=255, rng=rng)

        # Compose output: frozen layer + live layer
        live_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_live_tracks_rgba(live_layer, tracks, now)

        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.alpha_composite(acc_fill)
        out.alpha_composite(acc_lines)
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