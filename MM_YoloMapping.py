import os
import time
import random
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

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

def intersection(a: Box, b: Box) -> Box:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return Box(0, 0, 0, 0)
    return Box(x1, y1, x2, y2)

def cell_key(x1, y1, x2, y2):
    return (x1, y1, x2, y2)

def rect_contains_cell(r: Box, x1, y1, x2, y2) -> bool:
    return (r.x1 <= x1 and r.y1 <= y1 and r.x2 >= x2 and r.y2 >= y2)

def build_cells(xs, ys):
    xs = sorted(xs)
    ys = sorted(ys)
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if x2 <= x1:
            continue
        for j in range(len(ys) - 1):
            y1, y2 = ys[j], ys[j + 1]
            if y2 <= y1:
                continue
            yield (x1, y1, x2, y2)


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

def add_pairwise_intersections(boxes: List[Box], min_area: int = 200) -> List[Tuple[Box, str]]:
    items: List[Tuple[Box, str]] = [(b, "box") for b in boxes]
    seen = set(boxes)

    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            inter = intersection(boxes[i], boxes[j])
            if inter.area < min_area:
                continue
            if inter in seen:
                continue
            seen.add(inter)
            items.append((inter, "intersect"))
    return items

def center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    tw, th = img.size
    if target_w <= 0 or target_h <= 0:
        return img
    target_ar = target_w / target_h
    src_ar = tw / th

    if src_ar > target_ar:
        new_w = int(th * target_ar)
        left = (tw - new_w) // 2
        img = img.crop((left, 0, left + new_w, th))
    else:
        new_h = int(tw / target_ar)
        top = (th - new_h) // 2
        img = img.crop((0, top, tw, top + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

_cached_src = None
_cached_mtime = None

def load_random_fill_image(gen_folder: str) -> Image.Image | None:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(gen_folder) if f.lower().endswith(exts)]
    if not files:
        return None
    path = os.path.join(gen_folder, random.choice(files))
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None

def load_latest_fill_image(gen_folder: str) -> Image.Image | None:
    global _cached_src, _cached_mtime
    exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = [
        os.path.join(gen_folder, f)
        for f in os.listdir(gen_folder)
        if f.lower().endswith(exts)
    ]
    if not paths:
        return _cached_src  # reuse whatever we had

    newest = max(paths, key=lambda p: os.path.getmtime(p))
    mtime = os.path.getmtime(newest)

    if _cached_src is None or _cached_mtime != mtime:
        try:
            _cached_src = Image.open(newest).convert("RGBA")
            _cached_mtime = mtime
        except Exception:
            pass

    return _cached_src

def render_fill_layer(W, H, items, gen_folder):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    filled_keys = []

    for b, kind in items:
        if b.area == 0:
            continue

        src = load_random_fill_image(gen_folder)
        if src is None:
            continue

        tile = center_crop_to_aspect(src.convert("RGBA"), b.w, b.h)
        layer.alpha_composite(tile, dest=(b.x1, b.y1))

        filled_keys.append((b.x1, b.y1, b.x2, b.y2, kind))

    return layer, filled_keys



def render_outline_layer(W, H, items, w_box=6, w_inter=12):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for b, kind in items:
        if b.area == 0:
            continue
        width = w_inter if kind == "intersect" else w_box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(0,255,0,255), width=width)
    return layer

def draw_outlines_in_place(img: Image.Image, items: List[Tuple[Box, str]], w_box=3, w_inter=5, alpha=220):
    draw = ImageDraw.Draw(img)
    for b, kind in items:
        if b.area == 0:
            continue
        width = w_inter if kind == "intersect" else w_box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(0, 255, 0, alpha), width=width)

def pick_edge_pair(vals, min_span):
    """
    Pick two distinct edges from vals (sorted list), not necessarily adjacent.
    Biased toward shorter spans but still allows long spans.
    Returns (a, b) with b > a, or None.
    """
    if len(vals) < 2:
        return None
    n = len(vals)

    i = random.randrange(0, n - 1)

    # geometric-ish bias: small jumps common, big jumps rare
    jump = 1
    while i + jump < n and random.random() < 0.65:
        jump += 1
    j = min(n - 1, i + jump)

    a, b = vals[i], vals[j]
    if (b - a) < min_span:
        # try a few bigger jumps
        for _ in range(8):
            j = random.randrange(i + 1, n)
            a, b = vals[i], vals[j]
            if (b - a) >= min_span:
                break
        if (b - a) < min_span:
            return None
    return a, b

def jitter_edge(v, jitter_px, lo, hi):
    if jitter_px <= 0:
        return v
    return max(lo, min(hi, v + random.randint(-jitter_px, jitter_px)))

def add_history_intersections(
    new_boxes: List[Box],
    history: List[Box],
    min_area: int = 200
) -> List[Box]:
    inters: List[Box] = []
    for nb in new_boxes:
        for hb in history:
            inter = intersection(nb, hb)
            if inter.area >= min_area:
                inters.append(inter)
    return inters

def main():
    gen_folder = "/Users/j_laptop/mementomori/gen_images"
    out_path = "/Users/j_laptop/mementomori/accumulated.png"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    acc_fill = None
    acc_lines = None

    # arrangement state
    accepted_rects: List[Box] = []
    known_rects = set()          # geom_key(Box)
    xs, ys = set(), set()
    filled_cells = set()         # cell_key(x1,y1,x2,y2)

    # tuning
    Q = 24
    CONF = 0.25
    SLEEP_S = 4.0
    MAX_CELL_FILLS_PER_TICK = 14
    CELL_MIN_W = 40
    CELL_MIN_H = 40

    # keep things bounded
    MAX_ACCEPTED_RECTS = 400     # affects cell count; raise carefully
    MAX_EDGES = 250              # caps xs and ys sizes

    def accept_rect(b: Box):
        k = geom_key(b)
        if k in known_rects:
            return False
        known_rects.add(k)
        accepted_rects.append(b)
        xs.update([b.x1, b.x2])
        ys.update([b.y1, b.y2])
        return True

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        H, W, _ = frame.shape

        if acc_fill is None or acc_fill.size != (W, H):
            acc_fill = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            acc_lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            accepted_rects.clear()
            known_rects.clear()
            xs.clear()
            ys.clear()
            filled_cells.clear()

        results = model(frame)
        persons = results.xyxy[0].cpu().numpy()

        # 1) accept new YOLO boxes (these define the arrangement)
        boxes_now = boxes_from_yolo(persons, W, H, conf_thresh=CONF, q=Q)
        new_accepts = 0
        for b in boxes_now:
            if accept_rect(b):
                new_accepts += 1

        # 2) bound accepted rects and edge sets for performance
        if len(accepted_rects) > MAX_ACCEPTED_RECTS:
            # drop oldest; rebuild edges and known set from remaining
            accepted_rects[:] = accepted_rects[-MAX_ACCEPTED_RECTS:]
            known_rects = set(geom_key(r) for r in accepted_rects)
            xs = set()
            ys = set()
            for r in accepted_rects:
                xs.update([r.x1, r.x2])
                ys.update([r.y1, r.y2])

        if len(xs) > MAX_EDGES:
            xs = set(sorted(xs)[-MAX_EDGES:])
        if len(ys) > MAX_EDGES:
            ys = set(sorted(ys)[-MAX_EDGES:])

        # 3) build candidate cells and fill a few new ones
        to_fill: List[Tuple[Box, str]] = []
        for (x1, y1, x2, y2) in build_cells(xs, ys):
            if (x2 - x1) < CELL_MIN_W or (y2 - y1) < CELL_MIN_H:
                continue

            ck = cell_key(x1, y1, x2, y2)
            if ck in filled_cells:
                continue

            # coverage: cell must be inside at least one accepted rect
            covered = False
            for r in accepted_rects:
                if rect_contains_cell(r, x1, y1, x2, y2):
                    covered = True
                    break
            if not covered:
                continue

            to_fill.append((Box(x1, y1, x2, y2), "cell"))
            if len(to_fill) >= MAX_CELL_FILLS_PER_TICK:
                break

        fill_layer, filled_keys = render_fill_layer(W, H, to_fill, gen_folder=gen_folder)
        acc_fill.alpha_composite(fill_layer)

        # 4) persist outlines for newly filled cells
        new_for_lines: List[Tuple[Box, str]] = []
        for (x1, y1, x2, y2, _kind) in filled_keys:
            filled_cells.add(cell_key(x1, y1, x2, y2))
            new_for_lines.append((Box(x1, y1, x2, y2), "cell"))

        # thin outlines
        draw_outlines_in_place(acc_lines, new_for_lines, w_box=2, w_inter=2, alpha=220)

        out = acc_fill.copy()
        out.alpha_composite(acc_lines)
        out.save(out_path)

        print(
            "yolo_boxes", len(boxes_now),
            "new_accepts", new_accepts,
            "accepted_rects", len(accepted_rects),
            "edges", (len(xs), len(ys)),
            "filled_cells_total", len(filled_cells),
            "filled_this_tick", len(filled_keys),
        )

        time.sleep(SLEEP_S)





if __name__ == "__main__":
    main()
