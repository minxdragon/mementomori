import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw

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

@dataclass
class Track:
    track_id: int
    bbox: Box
    created_at: float
    last_seen_at: float
    frozen: bool = False
    frozen_box: Optional[Box] = None

def draw_live_tracks_rgba(live_layer: Image.Image, tracks: List[Track], now: float):
    draw = ImageDraw.Draw(live_layer)
    # neon-ish purple in RGB
    purple = (180, 0, 255, 255)
    for t in tracks:
        if t.frozen:
            continue
        age = max(0.0, now - t.created_at)
        # small pulse so it reads as "live"
        width = int(3 + 2 * (0.5 + 0.5 * np.sin(age * 8.0)))
        b = t.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=purple, width=width)

def stamp_frozen_green(accum: Image.Image, b: Box):
    draw = ImageDraw.Draw(accum)
    # filled green + slightly brighter outline so it stays readable
    fill = (0, 255, 0, 255)
    outline = (80, 255, 80, 255)
    draw.rectangle([b.x1, b.y1, b.x2, b.y2], fill=fill, outline=outline, width=2)

def main():
    base_dir = Path(os.path.expanduser("~/mementomori"))

    output_dir = base_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = base_dir / "latest.png"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    # Permanent layer where frozen rectangles live.
    acc_frozen: Optional[Image.Image] = None

    tracks: List[Track] = []
    next_id = 1
    frozen_keys = set()

    # Tuning
    Q = 24                 # quantize, makes boxes less jittery and more graphic
    CONF = 0.25
    DETECT_EVERY = 2       # run YOLO every N frames (keeps FPS sane)
    SMOOTH_T = 0.25        # 0..1, higher follows faster, lower smooths more
    LIVE_SECONDS = 1.4     # how long a box stays "live" before freezing
    MISS_SECONDS = 0.7     # how long we tolerate missing detections before dropping a live track
    IOU_THRESH = 0.20      # match threshold for associating detections with existing tracks

    SAVE_EVERY = 1         # save every N frames
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
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

        # 1) match detections to existing LIVE tracks (greedy IoU)
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
                if di in unmatched_det:
                    unmatched_det.remove(di)

        # 2) create new tracks for unmatched detections
        for di in sorted(unmatched_det):
            det = detections[di]
            tracks.append(Track(track_id=next_id, bbox=det, created_at=now, last_seen_at=now))
            next_id += 1

        # 3) drop stale live tracks (prevents ghosts)
        kept: List[Track] = []
        for t in tracks:
            if t.frozen:
                kept.append(t)
                continue
            if (now - t.last_seen_at) <= MISS_SECONDS:
                kept.append(t)
        tracks = kept

        # 4) freeze tracks that have existed long enough
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
                    stamp_frozen_green(acc_frozen, qb)

        # 5) render output: frozen layer + current live layer
        live_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_live_tracks_rgba(live_layer, tracks, now)

        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.alpha_composite(acc_frozen)
        out.alpha_composite(live_layer)

        if frame_idx % SAVE_EVERY == 0:
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            out.save(frame_path)

            tmp_latest = latest_path.with_suffix(".tmp.png")
            out.save(tmp_latest)
            os.replace(tmp_latest, latest_path)

        frame_idx += 1

        # Optional preview window for debugging (disable for install)
        # cv2.imshow("preview", cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGRA))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

if __name__ == "__main__":
    main()
