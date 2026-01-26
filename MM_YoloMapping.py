import os
import time
import random
from dataclasses import dataclass
from typing import List, Tuple

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
        if b.area > 0:
            out.append(b)
    return out

def add_pairwise_intersections(boxes: List[Box], min_area: int = 400) -> List[Tuple[Box, str]]:
    """
    Returns list of (box, kind) where kind is "box" or "intersect".
    """
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

def render_fill_layer(
    W: int,
    H: int,
    items: List[Tuple[Box, str]],
    gen_folder: str,
    tile_alpha_box: int = 140,
    tile_alpha_intersect: int = 220,  # intersections read stronger
) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for b, kind in items:
        if b.area == 0:
            continue

        src = load_random_fill_image(gen_folder)
        if src is None:
            continue

        tile = center_crop_to_aspect(src.convert("RGBA"), b.w, b.h)

        target_alpha = tile_alpha_intersect if kind == "intersect" else tile_alpha_box
        r, g, bl, a = tile.split()
        a = a.point(lambda _: target_alpha)
        tile = Image.merge("RGBA", (r, g, bl, a))

        layer.alpha_composite(tile, dest=(b.x1, b.y1))

    return layer


def render_outline_layer(
    W: int,
    H: int,
    items: List[Tuple[Box, str]],
    outline_width_box: int = 6,
    outline_width_intersect: int = 10,
) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for b, kind in items:
        if b.area == 0:
            continue

        width = outline_width_intersect if kind == "intersect" else outline_width_box
        color = (0, 255, 0, 255)  # fully opaque bright green

        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)

    return layer


def main():
    gen_folder = "/Users/j_laptop/mementomori/gen_images"
    out_path = "/Users/j_laptop/mementomori/accumulated.png"  # single file

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.eval()

    acc = None
    seen = set()  # comment out if you want to repaint every tick

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5)
            continue

        H, W, _ = frame.shape

        if acc is None:
            acc = Image.new("RGBA", (W, H), (0, 0, 0, 255))  # use (0,0,0,0) for transparent start

        results = model(frame)
        persons = results.xyxy[0].cpu().numpy()

        boxes = boxes_from_yolo(persons, W, H, conf_thresh=0.25, q=8)
        items = add_pairwise_intersections(boxes, min_area=800)

        # 1) fill only new rects (both box + intersect)
        new_items = []
        for b, kind in items:
            key = (b.x1, b.y1, b.x2, b.y2, kind)
            if key in seen:
                continue
            seen.add(key)
            new_items.append((b, kind))

        fill_layer = render_fill_layer(W, H, new_items, gen_folder=gen_folder)

        # 2) outlines for current intersections every tick (always bright, always on top)
        current_intersections = [(b, kind) for (b, kind) in items if kind == "intersect"]
        outline_layer = render_outline_layer(W, H, current_intersections)

        acc.alpha_composite(fill_layer)
        acc.alpha_composite(outline_layer)  # last write wins, keeps lines bright
        acc.save(out_path)


        time.sleep(10.0)


if __name__ == "__main__":
    main()
