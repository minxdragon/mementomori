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
        if b.w < 60 or b.h < 60:
            continue
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

def render_fill_layer(W, H, items, gen_folder):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for b, kind in items:
        if b.area == 0:
            continue
        src = load_random_fill_image(gen_folder)
        if src is None:
            continue
        tile = center_crop_to_aspect(src.convert("RGBA"), b.w, b.h)
        layer.alpha_composite(tile, dest=(b.x1, b.y1))

    return layer


def render_outline_layer(W, H, items, w_box=6, w_inter=12):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for b, kind in items:
        if b.area == 0:
            continue
        width = w_inter if kind == "intersect" else w_box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(0,255,0,255), width=width)
    return layer

def draw_outlines_in_place(img: Image.Image, items: List[Tuple[Box, str]], w_box=6, w_inter=14):
    draw = ImageDraw.Draw(img)
    for b, kind in items:
        if b.area == 0:
            continue
        width = w_inter if kind == "intersect" else w_box

        # black under-stroke
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(0, 0, 0, 255), width=width + 2)
        # green stroke
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(0, 255, 0, 255), width=width)

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
    seen = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        H, W, _ = frame.shape

        if acc_fill is None or acc_fill.size != (W, H):
            acc_fill = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            acc_lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            seen = set()

        results = model(frame)
        persons = results.xyxy[0].cpu().numpy()

        boxes = boxes_from_yolo(persons, W, H, conf_thresh=0.25, q=24)
        items = add_pairwise_intersections(boxes, min_area=800)

        # new rects only (boxes + intersections)
        new_items = []
        for b, kind in items:
            key = (b.x1, b.y1, b.x2, b.y2, kind)
            if key in seen:
                continue
            seen.add(key)
            new_items.append((b, kind))

        # fill only new rects
        fill_layer = render_fill_layer(W, H, new_items, gen_folder=gen_folder)
        acc_fill.alpha_composite(fill_layer)

        # persist outlines for new rects
        draw_outlines_in_place(acc_lines, new_items, w_box=6, w_inter=14)

        # output: fills + all past outlines
        out = acc_fill.copy()
        out.alpha_composite(acc_lines)
        out.save(out_path)

        time.sleep(10.0)



if __name__ == "__main__":
    main()
