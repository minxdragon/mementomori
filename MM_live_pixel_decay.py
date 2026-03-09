import os
import time
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import torch


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


@dataclass
class Track:
    track_id: int
    bbox: Box
    created_at: float
    last_seen_at: float
    frozen: bool = False


@dataclass
class Tile:
    bbox: Box
    kind: str  # 'box' or 'overlap'
    patch_original: Image.Image
    patch_current: Image.Image
    alpha: int
    decay_step: int = 0
    last_decay_at: float = 0.0


@dataclass
class Config:
    Q: int = 20
    CONF: float = 0.25
    DETECT_EVERY: int = 4
    DETECT_SCALE: float = 0.5
    SMOOTH_T: float = 0.40
    LIVE_SECONDS: float = 1.1
    MISS_SECONDS: float = 1.2
    IOU_THRESH: float = 0.12

    PURPLE_BGRA: Tuple[int, int, int, int] = (255, 0, 255, 255)
    GREEN_BGRA: Tuple[int, int, int, int] = (0, 255, 0, 255)
    LIVE_THICK: int = 3
    FROZEN_THICK_OUTER: int = 7
    FROZEN_THICK_INNER: int = 2

    BOX_ALPHA: int = 235
    OVERLAP_ALPHA: int = 235

    BOX_DECAY_INTERVAL: float = 5.0
    OVERLAP_DECAY_INTERVAL: float = 2.0
    BOX_DECAY_FACTORS: Tuple[int, ...] = (1, 2, 4, 8)
    OVERLAP_DECAY_FACTORS: Tuple[int, ...] = (1, 3, 6, 10, 16)

    SAVE_INTERVAL: float = 1.5
    SAVE_NUMBERED_FRAMES: bool = True

    MODEL_NAME: str = "yolov5n"
    MODEL_CONF: float = 0.35
    MODEL_IOU: float = 0.40
    MODEL_MAX_DET: int = 6


def clamp_box(b: Box, W: int, H: int) -> Box:
    x1 = max(0, min(W, b.x1))
    y1 = max(0, min(H, b.y1))
    x2 = max(0, min(W, b.x2))
    y2 = max(0, min(H, b.y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Box(x1, y1, x2, y2)


def quantize_box(b: Box, Q: int) -> Box:
    q0 = lambda v: (v // Q) * Q
    q1 = lambda v: ((v + Q - 1) // Q) * Q
    return Box(q0(b.x1), q0(b.y1), q1(b.x2), q1(b.y2))


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
    union = a.area + b.area - inter
    return float(inter) / float(union) if union > 0 else 0.0


def lerp_box(a: Box, b: Box, t: float) -> Box:
    return Box(
        int(round(a.x1 + (b.x1 - a.x1) * t)),
        int(round(a.y1 + (b.y1 - a.y1) * t)),
        int(round(a.x2 + (b.x2 - a.x2) * t)),
        int(round(a.y2 + (b.y2 - a.y2) * t)),
    )


def geom_key(b: Box):
    return (b.x1, b.y1, b.x2, b.y2)


def pil_rgba_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def seed_from_rect(tag: str, b: Box) -> int:
    key = f"{tag}:{b.x1},{b.y1},{b.x2},{b.y2}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def intersection_box(a: Box, b: Box) -> Optional[Box]:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def load_nature_images(nature_dir: Path) -> List[Image.Image]:
    imgs = []
    if not nature_dir.exists():
        return imgs
    for p in sorted(nature_dir.glob("*")):
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs


def random_nature_patch(nature_imgs: List[Image.Image], w: int, h: int, rng: random.Random) -> Image.Image:
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    if not nature_imgs:
        arr = np.frombuffer(rng.randbytes(w * h), dtype=np.uint8).reshape((h, w))
        rgb = np.stack([arr, arr, arr], axis=2)
        return Image.fromarray(rgb, "RGB").convert("RGBA")
    src = rng.choice(nature_imgs)
    sw, sh = src.size
    scale = max(w / sw, h / sh, 0.01)
    rw, rh = int(sw * scale + 0.5), int(sh * scale + 0.5)
    resized = src.resize((rw, rh), Image.Resampling.BICUBIC)
    x0 = 0 if rw == w else rng.randrange(0, max(1, rw - w))
    y0 = 0 if rh == h else rng.randrange(0, max(1, rh - h))
    return resized.crop((x0, y0, x0 + w, y0 + h)).convert("RGBA")


def draw_rect_rgba(img: Image.Image, b: Box, bgra: Tuple[int, int, int, int], thick: int):
    arr = np.array(img)
    b_, g_, r_, a_ = bgra
    color = (r_, g_, b_, a_)
    cv2.rectangle(arr, (b.x1, b.y1), (b.x2, b.y2), color, thickness=thick, lineType=cv2.LINE_AA)
    img.paste(Image.fromarray(arr, "RGBA"))


def stamp_frozen_outline(lines: Image.Image, b: Box, cfg: Config):
    draw_rect_rgba(lines, b, cfg.GREEN_BGRA, cfg.FROZEN_THICK_OUTER)
    draw_rect_rgba(lines, b, cfg.GREEN_BGRA, cfg.FROZEN_THICK_INNER)


def draw_live_tracks(live: Image.Image, tracks: List[Track], cfg: Config):
    for t in tracks:
        if t.frozen:
            continue
        b = quantize_box(t.bbox, cfg.Q)
        draw_rect_rgba(live, b, cfg.PURPLE_BGRA, cfg.LIVE_THICK)


def boxes_from_yolo_xyxy(persons_xyxy: np.ndarray, W: int, H: int, conf_thresh: float) -> List[Box]:
    out = []
    for row in persons_xyxy:
        x1, y1, x2, y2, conf, cls = row[:6]
        if float(conf) < conf_thresh:
            continue
        if int(cls) != 0:
            continue
        b = clamp_box(Box(int(x1), int(y1), int(x2), int(y2)), W, H)
        if b.area > 0:
            out.append(b)
    return out


def make_patch_for_box(b: Box, nature_imgs: List[Image.Image], alpha: int, tag: str) -> Image.Image:
    rng = random.Random(seed_from_rect(tag, b))
    patch = random_nature_patch(nature_imgs, b.w, b.h, rng)
    patch.putalpha(alpha)
    return patch


def pixelate_patch(patch: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return patch.copy()
    w, h = patch.size
    sw = max(1, w // factor)
    sh = max(1, h // factor)
    small = patch.resize((sw, sh), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def current_factor(tile: Tile, cfg: Config) -> int:
    factors = cfg.OVERLAP_DECAY_FACTORS if tile.kind == "overlap" else cfg.BOX_DECAY_FACTORS
    idx = min(tile.decay_step, len(factors) - 1)
    return factors[idx]


def decay_interval(tile: Tile, cfg: Config) -> float:
    return cfg.OVERLAP_DECAY_INTERVAL if tile.kind == "overlap" else cfg.BOX_DECAY_INTERVAL


def maybe_decay_tile(tile: Tile, now: float, cfg: Config) -> bool:
    interval = decay_interval(tile, cfg)
    if now - tile.last_decay_at < interval:
        return False

    factors = cfg.OVERLAP_DECAY_FACTORS if tile.kind == "overlap" else cfg.BOX_DECAY_FACTORS
    if tile.decay_step >= len(factors) - 1:
        tile.last_decay_at = now
        return False

    tile.decay_step += 1
    tile.last_decay_at = now
    factor = current_factor(tile, cfg)
    tile.patch_current = pixelate_patch(tile.patch_original, factor)
    return True


def composite_fill(fill_size: Tuple[int, int], tiles: List[Tile]) -> Image.Image:
    canvas = Image.new("RGBA", fill_size, (0, 0, 0, 0))
    for tile in tiles:
        canvas.alpha_composite(tile.patch_current, (tile.bbox.x1, tile.bbox.y1))
    return canvas


def main():
    cfg = Config()

    base_dir = Path(os.path.expanduser("~/mementomori"))
    base_dir.mkdir(parents=True, exist_ok=True)
    nature_imgs = load_nature_images(base_dir / "nature")

    output_dir = base_dir / "frames_pixel_decay"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = base_dir / "latest_pixel_decay.png"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.hub.load("ultralytics/yolov5", cfg.MODEL_NAME)
    model.to(device)
    model.eval()
    model.conf = cfg.MODEL_CONF
    model.iou = cfg.MODEL_IOU
    model.max_det = cfg.MODEL_MAX_DET

    acc_lines = None
    acc_fill = None

    tracks: List[Track] = []
    next_id = 1

    frozen_keys = set()
    overlap_keys = set()
    frozen_boxes: List[Box] = []
    tiles: List[Tile] = []
    latest_by_key: Dict[Tuple[str, int, int, int, int], Tile] = {}

    WIN = "mementomori_pixel_decay"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0
    saved_idx = 0
    last_save_at = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            now = time.time()
            H, W = frame.shape[:2]

            if acc_lines is None or acc_fill is None or acc_lines.size != (W, H) or acc_fill.size != (W, H):
                acc_lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                acc_fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))

                tracks.clear()
                next_id = 1
                frozen_keys.clear()
                overlap_keys.clear()
                frozen_boxes.clear()
                tiles.clear()
                latest_by_key.clear()

            detections = []
            if frame_idx % cfg.DETECT_EVERY == 0:
                if cfg.DETECT_SCALE != 1.0:
                    small = cv2.resize(frame, (0, 0), fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE)
                    results = model(small)
                    persons = results.xyxy[0].cpu().numpy()
                    if len(persons) > 0:
                        persons[:, 0:4] /= cfg.DETECT_SCALE
                else:
                    results = model(frame)
                    persons = results.xyxy[0].cpu().numpy()
                detections = boxes_from_yolo_xyxy(persons, W, H, cfg.CONF)

            unmatched = set(range(len(detections)))
            live_idx = [i for i, t in enumerate(tracks) if not t.frozen]
            used = set()

            for di, det in enumerate(detections):
                best_i = -1
                best_s = 0.0
                for ti in live_idx:
                    if ti in used:
                        continue
                    s = iou(det, tracks[ti].bbox)
                    if s > best_s:
                        best_s = s
                        best_i = ti
                if best_i >= 0 and best_s >= cfg.IOU_THRESH:
                    t = tracks[best_i]
                    tracks[best_i].bbox = lerp_box(t.bbox, det, cfg.SMOOTH_T)
                    tracks[best_i].last_seen_at = now
                    used.add(best_i)
                    unmatched.discard(di)

            for di in sorted(unmatched):
                det = detections[di]
                tracks.append(Track(next_id, det, now, now))
                next_id += 1

            tracks = [t for t in tracks if t.frozen or (now - t.last_seen_at) <= cfg.MISS_SECONDS]

            scene_changed = False
            new_frozen: List[Box] = []

            for t in tracks:
                if t.frozen:
                    continue
                if (now - t.created_at) >= cfg.LIVE_SECONDS:
                    t.frozen = True
                    qb = quantize_box(clamp_box(t.bbox, W, H), cfg.Q)
                    k = geom_key(qb)
                    if qb.area <= 0 or k in frozen_keys:
                        continue

                    frozen_keys.add(k)
                    frozen_boxes.append(qb)
                    new_frozen.append(qb)
                    stamp_frozen_outline(acc_lines, qb, cfg)

                    patch = make_patch_for_box(qb, nature_imgs, cfg.BOX_ALPHA, "box")
                    tile = Tile(
                        bbox=qb,
                        kind="box",
                        patch_original=patch.copy(),
                        patch_current=patch,
                        alpha=cfg.BOX_ALPHA,
                        decay_step=0,
                        last_decay_at=now,
                    )
                    tiles.append(tile)
                    latest_by_key[("box",) + geom_key(qb)] = tile
                    scene_changed = True

            if new_frozen:
                prior_boxes = frozen_boxes[:-len(new_frozen)] if len(new_frozen) <= len(frozen_boxes) else []
                for nb in new_frozen:
                    for ob in prior_boxes:
                        ib = intersection_box(nb, ob)
                        if ib is None:
                            continue
                        ib = quantize_box(clamp_box(ib, W, H), cfg.Q)
                        if ib.area <= 0:
                            continue

                        ik = geom_key(ib)
                        if ik in overlap_keys:
                            continue

                        overlap_keys.add(ik)
                        patch = make_patch_for_box(ib, nature_imgs, cfg.OVERLAP_ALPHA, "overlap")
                        tile = Tile(
                            bbox=ib,
                            kind="overlap",
                            patch_original=patch.copy(),
                            patch_current=patch,
                            alpha=cfg.OVERLAP_ALPHA,
                            decay_step=0,
                            last_decay_at=now,
                        )
                        tiles.append(tile)
                        latest_by_key[("overlap",) + geom_key(ib)] = tile
                        scene_changed = True

            any_decay = False
            for tile in tiles:
                if maybe_decay_tile(tile, now, cfg):
                    any_decay = True

            if scene_changed or any_decay:
                acc_fill = composite_fill((W, H), tiles)

            live_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_live_tracks(live_layer, tracks, cfg)

            out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            out.alpha_composite(acc_fill)
            out.alpha_composite(acc_lines)
            out.alpha_composite(live_layer)

            cv2.imshow(WIN, pil_rgba_to_bgr(out))

            if (now - last_save_at) >= cfg.SAVE_INTERVAL:
                last_save_at = now
                tmp = latest_path.with_suffix(".tmp.png")
                out.save(tmp)
                os.replace(tmp, latest_path)
                if cfg.SAVE_NUMBERED_FRAMES:
                    frame_path = output_dir / f"fixed_{saved_idx:06d}.png"
                    out.save(frame_path)
                    saved_idx += 1

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
