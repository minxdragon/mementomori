import os
import time
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

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
    yolo_id: Optional[int] = None


@dataclass
class Tile:
    bbox: Box
    patch_original: Image.Image
    patch_current: Image.Image
    created_at: float
    decay_step: float = 0.0
    last_decay_at: float = 0.0

@dataclass
class FreshStamp:
    bbox: Box
    patch: Image.Image
    created_at: float

@dataclass
class Config:
    Q: int = 4 #lower = more chaos and less orderly. increase to 20 to make it more grid-like
    CONF: float = 0.25
    DETECT_EVERY: int = 4
    DETECT_SCALE: float = 0.5
    SMOOTH_T: float = 0.50 #how much to interpolate bbox movement. increase to make it smoother but more laggy, decrease to make it more responsive but more jittery defaults to 0.40 which is a good balance for 30fps video
    LIVE_SECONDS: float = 4 #how long to keep a track as "live" before freezing it. increase to make it more forgiving of short occlusions, decrease to make it freeze faster
    MISS_SECONDS: float = 1.4 #how long to keep "live" tracks around without seeing them again before forgetting them. increase to make it more forgiving of missed detections, decrease to make it more responsive to change
    IOU_THRESH: float = 0.12

    PURPLE_BGRA: Tuple[int, int, int, int] = (255, 0, 255, 255)
    GREEN_BGRA: Tuple[int, int, int, int] = (0, 255, 0, 255)
    LIVE_THICK: int = 4
    FROZEN_THICK_OUTER: int = 2
    FROZEN_THICK_INNER: int = 2

    BOX_ALPHA: int = 235
    BOX_DECAY_INTERVAL: float = 5.0
    BOX_DECAY_FACTORS: Tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    
    TILE_FADE_START: float = 4.0
    TILE_FADE_DURATION: float = 4.0
    TILE_MIN_ALPHA: float = 0.04

    FRESH_STAMP_DURATION: float = 8.0
    FRESH_STAMP_ALPHA: int = 255
    MAX_FRESH_STAMPS: int = 200  # maximum number of fresh stamps to keep in memory
    MAX_TILES: int = 300  # maximum number of tile objects to keep in memory

    SAVE_INTERVAL: float = 1.5
    SAVE_NUMBERED_FRAMES: bool = True

    MODEL_NAME: str = "yolov8n"
    MODEL_CONF: float = 0.35
    MODEL_IOU: float = 0.40
    MODEL_MAX_DET: int = 3 #increase to allow more people tracking, but beware of performance impact

    GENERATED_DIR: str = "generated"

    GEN_START_INTERVAL: int = 30      # start with a new generation every N captures
    GEN_INTERVAL_SHRINK: int = 1      # reduce interval by this amount
    GEN_INTERVAL_STEP: int = 30       # every Y captures shrink interval
    GEN_MIN_INTERVAL: int = 1         # never go below this


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


def random_nature_patch(nature_imgs: List[Image.Image], w: int, h: int, rng: random.Random, used_indices: set = None) -> Image.Image:
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    if not nature_imgs:
        arr = np.frombuffer(rng.randbytes(w * h), dtype=np.uint8).reshape((h, w))
        rgb = np.stack([arr, arr, arr], axis=2)
        return Image.fromarray(rgb, "RGB").convert("RGBA")
    
    # Track used images to ensure each appears only once
    if used_indices is None:
        used_indices = set()
    
    available_indices = [i for i in range(len(nature_imgs)) if i not in used_indices]
    if not available_indices:
        # Reset if all used
        used_indices.clear()
        available_indices = list(range(len(nature_imgs)))
    
    idx = rng.choice(available_indices)
    used_indices.add(idx)
    src = nature_imgs[idx]
    
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


def make_patch_for_box(b: Box, nature_imgs: List[Image.Image], alpha: int, tag: str, used_indices: set = None) -> Image.Image:
    rng = random.Random(seed_from_rect(tag, b))
    patch = random_nature_patch(nature_imgs, b.w, b.h, rng, used_indices)
    patch.putalpha(alpha)
    return patch


def pixelate_patch(patch: Image.Image, factor: float) -> Image.Image:
    try:
        if factor <= 1:
            return patch.copy()
        w, h = patch.size
        sw = max(1, int(w // factor))
        sh = max(1, int(h // factor))
        small = patch.resize((sw, sh), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.NEAREST)
    except Exception as e:
        print(f"Error in pixelate_patch (factor={factor}, patch_size={patch.size}): {e}")
        return patch.copy()

def maybe_decay_tile(tile: Tile, now: float, cfg: Config) -> bool:
    try:
        if now - tile.last_decay_at < cfg.BOX_DECAY_INTERVAL:
            return False
        if tile.decay_step >= cfg.BOX_DECAY_FACTORS[-1]: 
            tile.last_decay_at = now
            return False
        tile.decay_step += 0.5 # increase decay step to create a smoother fade effect (default 1.0 creates a more abrupt pixelation)
        tile.last_decay_at = now
        tile.patch_current = pixelate_patch(tile.patch_original, tile.decay_step)
        return True
    except Exception as e:
        print(f"Error in maybe_decay_tile (decay_step={tile.decay_step}): {e}")
        return False


def composite_fill(fill_size: Tuple[int, int], tiles: List[Tile]) -> Image.Image:
    try:
        canvas = Image.new("RGBA", fill_size, (0, 0, 0, 0))

        for tile in tiles:
            try:
                canvas.alpha_composite(tile.patch_current, (tile.bbox.x1, tile.bbox.y1))
            except Exception as e:
                print(f"Error compositing tile at ({tile.bbox.x1}, {tile.bbox.y1}): {e}")

        return canvas
    except Exception as e:
        print(f"Error in composite_fill: {e}")
        return Image.new("RGBA", fill_size, (0, 0, 0, 0))


def prune_tiles_fresh_stamps(tiles: List[Tile], fresh_stamps: List[FreshStamp], frozen_outlines: List[Tuple[Box, float]], acc_lines: Image.Image, cfg: Config) -> Image.Image:
    if len(tiles) > cfg.MAX_TILES:
        tiles.sort(key=lambda t: t.created_at)
        remove_count = len(tiles) - cfg.MAX_TILES
        if remove_count > 0:
            del tiles[:remove_count]
            print(f"Pruned {remove_count} oldest tiles")

    if len(fresh_stamps) > cfg.MAX_FRESH_STAMPS:
        fresh_stamps.sort(key=lambda s: s.created_at)
        remove_count = len(fresh_stamps) - cfg.MAX_FRESH_STAMPS
        if remove_count > 0:
            del fresh_stamps[:remove_count]
            print(f"Pruned {remove_count} oldest fresh stamps")

    if len(frozen_outlines) > cfg.MAX_TILES:
        frozen_outlines.sort(key=lambda f: f[1])
        remove_count = len(frozen_outlines) - cfg.MAX_TILES
        if remove_count > 0:
            del frozen_outlines[:remove_count]
            print(f"Pruned {remove_count} oldest outlines")
            # rebuild outlines layer
            acc_lines = Image.new("RGBA", acc_lines.size, (0, 0, 0, 0))
            for outline_box, _ in frozen_outlines:
                stamp_frozen_outline(acc_lines, outline_box, cfg)

    return acc_lines


def fresh_stamp_alpha(stamp: FreshStamp, now: float, cfg: Config) -> float:
    age = now - stamp.created_at
    if age <= 0:
        return 1.0
    t = min(1.0, age / cfg.FRESH_STAMP_DURATION)
    return 1.0 - t


def composite_fresh_stamps(
    size: Tuple[int, int],
    fresh_stamps: List[FreshStamp],
    now: float,
    cfg: Config
) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))

    for stamp in fresh_stamps:
        a = fresh_stamp_alpha(stamp, now, cfg)
        if a <= 0:
            continue

        patch = stamp.patch.copy()
        arr = np.array(patch)
        arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * a).clip(0, 255).astype(np.uint8)
        patch = Image.fromarray(arr, "RGBA")

        layer.alpha_composite(patch, (stamp.bbox.x1, stamp.bbox.y1))

        outline_alpha = int(255 * a)
        draw_rect_rgba(layer, stamp.bbox, (0, 255, 0, outline_alpha), cfg.FROZEN_THICK_OUTER)
        draw_rect_rgba(layer, stamp.bbox, (0, 255, 0, outline_alpha), cfg.FROZEN_THICK_INNER)

    return layer

stamped_ids = set()
def main():
    cfg = Config()

    base_dir = Path(os.path.expanduser("~/mementomori"))
    base_dir.mkdir(parents=True, exist_ok=True)
    nature_imgs = load_nature_images(base_dir / "nature")
    generated_imgs = load_nature_images(base_dir / cfg.GENERATED_DIR)

    output_dir = base_dir / "ACMI_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = base_dir / "ACMI.png"
    tiles: List[Tile] = []
    fresh_stamps: List[FreshStamp] = []
    frozen_outlines: List[Tuple[Box, float]] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    # model.to(device)
    # model.eval()
    # model.conf = cfg.MODEL_CONF
    # model.iou = cfg.MODEL_IOU
    # model.max_det = cfg.MODEL_MAX_DET

    acc_lines = None
    acc_fill = None

    tracks: List[Track] = []
    next_id = 1

    frozen_keys = set()
    frozen_yolo_ids = set()
    tiles: List[Tile] = []
    boxes = None
    used_nature_imgs = set()
    used_generated_imgs = set()

    #capture_count is used to determine when to increase the generation interval
    capture_count = 0
    gen_interval = cfg.GEN_START_INTERVAL
    next_gen_capture = gen_interval

    WIN = "memento_mori"
    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WIN, 0, 0)
    cv2.resizeWindow(WIN, 1920, 1080)

    frame_idx = 0
    saved_idx = 0
    last_save_at = 0.0
    next_interval_bump = cfg.GEN_INTERVAL_STEP

    try:
        while True:
                try:
                    ok, frame = cap.read()
                    if not ok:
                        print("Warning: Failed to read frame from camera")
                        time.sleep(0.1)
                        continue
                    frame = cv2.flip(frame, 1)   # horizontal mirror
                except Exception as e:
                    print(f"Error reading from camera: {e}")
                    print("Attempting to reconnect to camera...")
                    try:
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            print("Failed to reconnect to camera, retrying in 2 seconds...")
                            time.sleep(2)
                            continue
                        print("Camera reconnected successfully")
                    except Exception as reconnect_error:
                        print(f"Error during camera reconnection: {reconnect_error}")
                        time.sleep(2)
                    continue

                now = time.time()
                H, W = frame.shape[:2]

                if acc_lines is None or acc_fill is None or acc_lines.size != (W, H) or acc_fill.size != (W, H):
                    acc_lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    acc_fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))

                    tracks.clear()
                    next_id = 1
                    frozen_keys.clear()
                    tiles.clear()
                    fresh_stamps.clear()
                    frozen_yolo_ids.clear()
                    used_nature_imgs.clear()
                    used_generated_imgs.clear()

                detections = []

                if frame_idx % cfg.DETECT_EVERY == 0:
                    try:
                        if cfg.DETECT_SCALE != 1.0:
                            small = cv2.resize(frame, (0, 0), fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE)
                        else:
                            small = frame

                        results = model.track(
                            source=small,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=[0],
                            conf=cfg.MODEL_CONF,
                            iou=cfg.MODEL_IOU,
                            verbose=False,
                            device=device
                        )
                        r = results[0]
                        boxes = r.boxes
                        print("detections this pass:", 0 if boxes is None else len(boxes))
                    except Exception as e:
                        print(f"Warning: YOLO detection failed (possible system overload): {e}")
                        boxes = None

                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()

                        if boxes.id is not None:
                            ids = boxes.id.cpu().numpy().astype(int)
                        else:
                            ids = np.full(len(xyxy), -1, dtype=int)

                        for box, c, det_yolo_id in zip(xyxy, cls, ids):
                            if int(c) != 0:
                                continue

                            x1, y1, x2, y2 = box
                            if cfg.DETECT_SCALE != 1.0:
                                x1 /= cfg.DETECT_SCALE
                                y1 /= cfg.DETECT_SCALE
                                x2 /= cfg.DETECT_SCALE
                                y2 /= cfg.DETECT_SCALE

                            b = clamp_box(Box(int(x1), int(y1), int(x2), int(y2)), W, H)
                            if b.area <= 0:
                                continue

                            detections.append((b, None if det_yolo_id == -1 else int(det_yolo_id)))

                unmatched = set(range(len(detections)))
                live_idx = [i for i, t in enumerate(tracks) if not t.frozen]
                used = set()

                for di, (det, det_yolo_id) in enumerate(detections):
                    best_i = -1

                    # first try exact YOLO ID match
                    if det_yolo_id is not None:
                        for ti in live_idx:
                            if ti in used:
                                continue
                            if tracks[ti].yolo_id == det_yolo_id:
                                best_i = ti
                                break

                    # fallback to IoU only if no ID match
                    if best_i == -1:
                        best_s = 0.0
                        for ti in live_idx:
                            if ti in used:
                                continue
                            s = iou(det, tracks[ti].bbox)
                            if s > best_s:
                                best_s = s
                                best_i = ti if s >= cfg.IOU_THRESH else -1

                    if best_i >= 0:
                        t = tracks[best_i]
                        tracks[best_i].bbox = lerp_box(t.bbox, det, cfg.SMOOTH_T)
                        tracks[best_i].last_seen_at = now
                        if det_yolo_id is not None:
                            tracks[best_i].yolo_id = det_yolo_id
                        used.add(best_i)
                        unmatched.discard(di)

                for di in sorted(unmatched):
                    det, det_yolo_id = detections[di]
                    tracks.append(Track(next_id, det, now, now, frozen=False, yolo_id=det_yolo_id))
                    next_id += 1

                tracks = [t for t in tracks if t.frozen or (now - t.last_seen_at) <= cfg.MISS_SECONDS]

                scene_changed = False
                
                for t in tracks:
                    if t.frozen:
                        continue
                    if (now - t.created_at) >= cfg.LIVE_SECONDS:
                        if t.yolo_id is not None and t.yolo_id in frozen_yolo_ids:
                            t.frozen = True
                            continue

                        t.frozen = True

                        if t.yolo_id is not None:
                            frozen_yolo_ids.add(t.yolo_id)

                        qb = quantize_box(clamp_box(t.bbox, W, H), cfg.Q)
                        k = geom_key(qb)

                        if qb.area <= 0 or k in frozen_keys:
                            continue

                        frozen_keys.add(k)
                        frozen_outlines.append((qb, now))
                        stamp_frozen_outline(acc_lines, qb, cfg)

                        patch = make_patch_for_box(qb, nature_imgs, cfg.BOX_ALPHA, "box", used_nature_imgs)
                        fresh_patch = patch.copy()
                        fresh_patch.putalpha(cfg.FRESH_STAMP_ALPHA)
                        tile = Tile(
                            bbox=qb,
                            patch_original=patch.copy(),
                            patch_current=patch,
                            created_at=now,
                            decay_step=0,
                            last_decay_at=now,
                        )

                        capture_count += 1

                        if capture_count >= next_interval_bump:
                            gen_interval = max(
                                cfg.GEN_MIN_INTERVAL,
                                gen_interval - cfg.GEN_INTERVAL_SHRINK
                            )
                            next_interval_bump += cfg.GEN_INTERVAL_STEP
                            print(f"Generation interval decreased to {gen_interval}")

                        if generated_imgs and capture_count >= next_gen_capture:
                            rng = random.Random(capture_count)

                            gen_patch = random_nature_patch(generated_imgs, qb.w, qb.h, rng, used_generated_imgs)
                            gen_patch.putalpha(cfg.BOX_ALPHA)
                            fresh_patch = gen_patch.copy()
                            fresh_patch.putalpha(cfg.FRESH_STAMP_ALPHA)

                            tile.patch_original = gen_patch.copy()
                            tile.patch_current = gen_patch
                            tile.created_at = now
                            tile.decay_step = 0
                            tile.last_decay_at = now

                            print(f"Replaced tile with generated image at capture {capture_count}")

                            next_gen_capture += gen_interval

                        tiles.append(tile)
                        fresh_stamps.append(FreshStamp(
                            bbox=qb,
                            patch=fresh_patch,
                            created_at=now,
                        ))

                        # Cap number of tiles + fresh stamps + outlines, deleting oldest when exceeded
                        acc_lines = prune_tiles_fresh_stamps(tiles, fresh_stamps, frozen_outlines, acc_lines, cfg)

                        scene_changed = True

                any_decay = False
                for tile in tiles:
                    if maybe_decay_tile(tile, now, cfg):
                        any_decay = True

                acc_fill = composite_fill((W, H), tiles)

                live_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                draw_live_tracks(live_layer, tracks, cfg)
                
                fresh_stamps = [
                    s for s in fresh_stamps
                    if (now - s.created_at) < cfg.FRESH_STAMP_DURATION
                ]

                acc_lines = prune_tiles_fresh_stamps(tiles, fresh_stamps, frozen_outlines, acc_lines, cfg)

                fresh_layer = composite_fresh_stamps((W, H), fresh_stamps, now, cfg)

                try:
                    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    out.alpha_composite(acc_fill)
                    out.alpha_composite(acc_lines)
                    out.alpha_composite(fresh_layer)
                    out.alpha_composite(live_layer)
                    
                    display_bgr = pil_rgba_to_bgr(out)
                    display_bgr = cv2.resize(display_bgr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(WIN, display_bgr)
                except Exception as e:
                    print(f"Warning: Display update failed (possible system overload): {e}")
                # print("actual capture:",
                # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

                if (now - last_save_at) >= cfg.SAVE_INTERVAL:
                    try:
                        last_save_at = now
                        tmp = latest_path.with_suffix(".tmp.png")
                        out.save(tmp)
                        os.replace(tmp, latest_path)
                        if cfg.SAVE_NUMBERED_FRAMES:
                            frame_path = output_dir / f"ACMI_{saved_idx:06d}.png"
                            out.save(frame_path)
                            saved_idx += 1
                    except Exception as e:
                        print(f"Warning: Save failed (possible disk full or permission issue): {e}")

                frame_idx += 1
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                except Exception as e:
                    print(f"Warning: Input handling error: {e}")
    except Exception as e:
        print(f"Critical error in main loop (system overload?): {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            print(f"Final stamp count: {len(tiles)}")
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
