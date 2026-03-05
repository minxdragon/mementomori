from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image

#core data
@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int
    @property
    def area(self): return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

@dataclass
class Track:
    track_id: int
    bbox: Box
    created_at: float
    last_seen_at: float
    frozen: bool = False
    frozen_box: Optional[Box] = None

@dataclass
class GridState:
    Q: int
    W: int = 0
    H: int = 0
    GW0: int = 0
    GH0: int = 0
    GW: int = 0
    GH: int = 0
    Hedge: Optional[np.ndarray] = None
    Vedge: Optional[np.ndarray] = None

    def reset(self, W: int, H: int):
        self.W, self.H = W, H
        self.GW0, self.GH0 = W // self.Q, H // self.Q
        self.GW, self.GH = self.GW0 + 2, self.GH0 + 2
        self.Hedge = np.zeros((self.GH + 1, self.GW), dtype=np.uint8)
        self.Vedge = np.zeros((self.GH, self.GW + 1), dtype=np.uint8)

@dataclass
class Layers:
    fill: Optional[Image.Image] = None
    lines: Optional[Image.Image] = None

    def reset(self, W: int, H: int):
        self.fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))

@dataclass
class State:
    Q: int
    layers: Layers = field(default_factory=Layers)
    grid: GridState = field(init=False)
    tracks: List[Track] = field(default_factory=list)
    frozen_boxes: List[Box] = field(default_factory=list)
    frozen_keys: set = field(default_factory=set)
    next_id: int = 1

    def __post_init__(self):
        self.grid = GridState(Q=self.Q)

    def reset(self, W: int, H: int):
        self.layers.reset(W, H)
        self.grid.reset(W, H)
        self.tracks.clear()
        self.frozen_boxes.clear()
        self.frozen_keys.clear()
        self.next_id = 1

#inferring geometry
def add_box_edges(grid: GridState, b: Box):
    Q = grid.Q
    GW0, GH0 = grid.GW0, grid.GH0

    gx0 = 1 + max(0, min(GW0, b.x1 // Q))
    gx1 = 1 + max(0, min(GW0, b.x2 // Q))
    gy0 = 1 + max(0, min(GH0, b.y1 // Q))
    gy1 = 1 + max(0, min(GH0, b.y2 // Q))
    if gx1 <= gx0 or gy1 <= gy0:
        return

    grid.Hedge[gy0, gx0:gx1] = 1
    grid.Hedge[gy1, gx0:gx1] = 1
    grid.Vedge[gy0:gy1, gx0] = 1
    grid.Vedge[gy0:gy1, gx1] = 1


def find_closed_rectangles_on_grid(grid: GridState) -> List[Tuple[int, int, int, int]]:
    from collections import deque
    Hedge, Vedge = grid.Hedge, grid.Vedge
    GH, GW, Q = grid.GH, grid.GW, grid.Q

    visited = np.zeros((GH - 2, GW - 2), dtype=np.uint8)
    rects = []

    def neighbors(y, x):
        gy, gx = y + 1, x + 1
        if y > 0 and Hedge[gy, gx] == 0: yield (y - 1, x)
        if y < (GH - 3) and Hedge[gy + 1, gx] == 0: yield (y + 1, x)
        if x > 0 and Vedge[gy, gx] == 0: yield (y, x - 1)
        if x < (GW - 3) and Vedge[gy, gx + 1] == 0: yield (y, x + 1)

    for sy in range(GH - 2):
        for sx in range(GW - 2):
            if visited[sy, sx]:
                continue

            q = deque([(sy, sx)])
            visited[sy, sx] = 1

            cells = 0
            minx = maxx = sx
            miny = maxy = sy

            touches_outer = (sx == 0 or sy == 0 or sx == (GW - 3) or sy == (GH - 3))

            while q:
                y, x = q.popleft()
                cells += 1
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                if x == 0 or y == 0 or x == (GW - 3) or y == (GH - 3):
                    touches_outer = True
                for ny, nx in neighbors(y, x):
                    if not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))

            if touches_outer:
                continue

            w = (maxx - minx + 1)
            h = (maxy - miny + 1)
            if cells != w * h:
                continue

            x1 = minx * Q
            y1 = miny * Q
            x2 = (maxx + 1) * Q
            y2 = (maxy + 1) * Q
            rects.append((x1, y1, x2, y2))

    return rects

#main loop
def tick(state: State, frame_bgr, now, model, nature_imgs, cfg):
    H, W = frame_bgr.shape[:2]
    if state.layers.fill is None or state.layers.fill.size != (W, H):
        state.reset(W, H)

    detections = run_yolo_every_n_frames(frame_bgr, model, cfg)
    update_tracks(state, detections, now, cfg)

    froze_any = False
    for t in state.tracks:
        if t.frozen:
            continue
        if (now - t.created_at) >= cfg.LIVE_SECONDS:
            t.frozen = True
            qb = quantize_box(clamp_box(t.bbox, W, H), state.Q)
            k = (qb.x1, qb.y1, qb.x2, qb.y2)
            if qb.area > 0 and k not in state.frozen_keys:
                state.frozen_keys.add(k)
                state.frozen_boxes.append(qb)

                stamp_frozen_outline(state.layers.lines, qb)
                add_box_edges(state.grid, qb)
                froze_any = True

    if froze_any:
        rects = find_closed_rectangles_on_grid(state.grid)
        repaint_fill(state.layers.fill, state.frozen_boxes, rects, nature_imgs)

    live = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_live_tracks(live, state.tracks, now)

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(state.layers.fill)
    out.alpha_composite(state.layers.lines)
    out.alpha_composite(live)

    return out, froze_any