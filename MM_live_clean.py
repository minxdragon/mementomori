# generated

import os, time, random, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Set

import cv2
import numpy as np
from PIL import Image
import torch

@dataclass(frozen=True)
class Box:
    x1:int; y1:int; x2:int; y2:int
    @property
    def w(self): return max(0, self.x2-self.x1)
    @property
    def h(self): return max(0, self.y2-self.y1)
    @property
    def area(self): return self.w*self.h

@dataclass
class Track:
    track_id:int
    bbox:Box
    created_at:float
    last_seen_at:float
    frozen:bool=False

@dataclass
class Config:
    Q:int=20 #uniformity. lower is less uniform, higher is more uniform
    CONF:float=0.45 #confidence threshold for person detection
    DETECT_EVERY:int=6 #how frequently to run detection (frames)
    SMOOTH_T:float=0.25
    LIVE_SECONDS:float=1.5 #tracking a person
    MISS_SECONDS:float=1.2 
    IOU_THRESH:float=0.20
    PURPLE_BGRA:Tuple[int,int,int,int]=(255,0,255,255)
    GREEN_BGRA:Tuple[int,int,int,int]=(0,255,0,255)
    LIVE_THICK:int=2
    FROZEN_THICK_OUTER:int=2
    FROZEN_THICK_INNER:int=2
    FILL_ALPHA:int=255
    MAX_CELLS = 150

patch_cache = {}

def clamp_box(b:Box, W:int, H:int)->Box:
    x1=max(0,min(W,b.x1)); y1=max(0,min(H,b.y1))
    x2=max(0,min(W,b.x2)); y2=max(0,min(H,b.y2))
    if x2<x1: x1,x2=x2,x1
    if y2<y1: y1,y2=y2,y1
    return Box(x1,y1,x2,y2)

def quantize_box(b:Box,Q:int)->Box:
    q0=lambda v:(v//Q)*Q
    q1=lambda v:((v+Q-1)//Q)*Q
    return Box(q0(b.x1),q0(b.y1),q1(b.x2),q1(b.y2))

def iou(a:Box,b:Box)->float:
    ix1=max(a.x1,b.x1); iy1=max(a.y1,b.y1)
    ix2=min(a.x2,b.x2); iy2=min(a.y2,b.y2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    union=a.area+b.area-inter
    return float(inter)/float(union) if union>0 else 0.0

def lerp_box(a:Box,b:Box,t:float)->Box:
    return Box(int(round(a.x1+(b.x1-a.x1)*t)),
               int(round(a.y1+(b.y1-a.y1)*t)),
               int(round(a.x2+(b.x2-a.x2)*t)),
               int(round(a.y2+(b.y2-a.y2)*t)))

def geom_key(b:Box): return (b.x1,b.y1,b.x2,b.y2)

def pil_rgba_to_bgr(img:Image.Image)->np.ndarray:
    arr=np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def seed_from_rect(tag:str,b:Box)->int:
    key=f"{tag}:{b.x1},{b.y1},{b.x2},{b.y2}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),"little")


def load_nature_images(nature_dir:Path)->List[Image.Image]:
    imgs=[]
    if not nature_dir.exists(): return imgs
    for p in sorted(nature_dir.glob("*")):
        if p.suffix.lower() not in [".png",".jpg",".jpeg",".webp",".bmp"]: continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs

def random_nature_patch(nature_imgs:List[Image.Image], w:int, h:int, rng:random.Random)->Image.Image:
    if w<=0 or h<=0:
        return Image.new("RGBA",(max(1,w),max(1,h)),(0,0,0,0))
    if not nature_imgs:
        # cheap noise fallback
        arr=np.frombuffer(rng.randbytes(w*h), dtype=np.uint8).reshape((h,w))
        rgb=np.stack([arr,arr,arr],axis=2)
        return Image.fromarray(rgb,"RGB").convert("RGBA")
    src=rng.choice(nature_imgs); sw,sh=src.size
    scale=max(w/sw, h/sh, 0.01)
    rw,rh=int(sw*scale+0.5), int(sh*scale+0.5)
    key=(id(src), rw, rh)
    if key not in patch_cache:
        patch_cache[key]=src.resize((rw,rh), Image.BICUBIC)
    resized=patch_cache[key]
    x0=0 if rw==w else rng.randrange(0, max(1,rw-w))
    y0=0 if rh==h else rng.randrange(0, max(1,rh-h))
    return resized.crop((x0,y0,x0+w,y0+h)).convert("RGBA")

def stamp_fill(acc_fill:Image.Image, b:Box, nature_imgs:List[Image.Image], fill_alpha:int, rng:random.Random):
    W,H=acc_fill.size
    b=clamp_box(b,W,H)
    if b.area<=0: return
    patch=random_nature_patch(nature_imgs, b.w, b.h, rng)
    patch.putalpha(fill_alpha)
    acc_fill.paste(patch,(b.x1,b.y1),patch)


def draw_rect_rgba(img:Image.Image, b:Box, bgra:Tuple[int,int,int,int], thick:int):
    arr=np.array(img)
    b_,g_,r_,a_=bgra
    color=(r_,g_,b_,a_)
    cv2.rectangle(arr,(b.x1,b.y1),(b.x2,b.y2),color,thickness=thick,lineType=cv2.LINE_AA)
    img.paste(Image.fromarray(arr,"RGBA"))

def stamp_frozen_outline(lines:Image.Image, b:Box, cfg:Config):
    draw_rect_rgba(lines,b,cfg.GREEN_BGRA,cfg.FROZEN_THICK_OUTER)
    draw_rect_rgba(lines,b,cfg.GREEN_BGRA,cfg.FROZEN_THICK_INNER)

def draw_live_tracks(live:Image.Image, tracks:List[Track], cfg:Config):
    for t in tracks:
        if t.frozen: continue
        b=quantize_box(t.bbox,cfg.Q)
        draw_rect_rgba(live,b,cfg.PURPLE_BGRA,cfg.LIVE_THICK)


@dataclass
class Grid:
    Q:int
    W:int=0; H:int=0
    GW0:int=0; GH0:int=0
    GW:int=0; GH:int=0
    Hedge:Optional[np.ndarray]=None
    Vedge:Optional[np.ndarray]=None
    def reset(self,W:int,H:int):
        self.W,self.H=W,H
        self.GW0,self.GH0=W//self.Q, H//self.Q
        self.GW,self.GH=self.GW0+2, self.GH0+2
        self.Hedge=np.zeros((self.GH+1,self.GW),dtype=np.uint8)
        self.Vedge=np.zeros((self.GH,self.GW+1),dtype=np.uint8)

def add_box_edges(grid:Grid, b:Box):
    Q=grid.Q
    gx0=1+max(0,min(grid.GW0,b.x1//Q))
    gx1=1+max(0,min(grid.GW0,b.x2//Q))
    gy0=1+max(0,min(grid.GH0,b.y1//Q))
    gy1=1+max(0,min(grid.GH0,b.y2//Q))
    if gx1<=gx0 or gy1<=gy0: return
    grid.Hedge[gy0,gx0:gx1]=1
    grid.Hedge[gy1,gx0:gx1]=1
    grid.Vedge[gy0:gy1,gx0]=1
    grid.Vedge[gy0:gy1,gx1]=1

def find_closed_rectangles(grid: Grid) -> List[Box]:
    Hedge = grid.Hedge
    Vedge = grid.Vedge
    Q = grid.Q

    GHc = grid.GH - 2
    GWc = grid.GW - 2

    rects = []

    for y0 in range(GHc):
        for y1 in range(y0 + 1, GHc + 1):

            # check horizontal edges exist
            if not np.any(Hedge[y0 + 1]): 
                continue
            if not np.any(Hedge[y1 + 1]):
                continue

            for x0 in range(GWc):
                for x1 in range(x0 + 1, GWc + 1):

                    if Hedge[y0 + 1, x0 + 1] == 0:
                        continue
                    if Hedge[y1 + 1, x0 + 1] == 0:
                        continue
                    if Hedge[y0 + 1, x1 + 1] == 0:
                        continue
                    if Hedge[y1 + 1, x1 + 1] == 0:
                        continue

                    if Vedge[y0 + 1, x0 + 1] == 0:
                        continue
                    if Vedge[y1 + 1, x0 + 1] == 0:
                        continue
                    if Vedge[y0 + 1, x1 + 1] == 0:
                        continue
                    if Vedge[y1 + 1, x1 + 1] == 0:
                        continue

                    rects.append(
                        Box(
                            x0 * Q,
                            y0 * Q,
                            x1 * Q,
                            y1 * Q
                        )
                    )

    return rects


def boxes_from_yolo_xyxy(persons_xyxy:np.ndarray, W:int, H:int, conf_thresh:float)->List[Box]:
    out=[]
    for row in persons_xyxy:
        x1,y1,x2,y2,conf,cls=row[:6]
        if float(conf)<conf_thresh: continue
        if int(cls)!=0: continue
        b=clamp_box(Box(int(x1),int(y1),int(x2),int(y2)),W,H)
        if b.area>0: out.append(b)
    return out


def main():
    cfg=Config()
    base_dir=Path(os.path.expanduser("~/mementomori"))
    base_dir.mkdir(parents=True, exist_ok=True)
    nature_imgs=load_nature_images(base_dir/"nature")
    output_dir=base_dir/"frames"; output_dir.mkdir(parents=True, exist_ok=True)
    latest_path=base_dir/"latest.png"

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model=torch.hub.load("ultralytics/yolov5","yolov5s")
    model.eval()
    #control for number of detections and speed
    model.conf = 0.45
    model.iou = 0.45
    model.max_det = 10


    acc_fill=None; acc_lines=None
    grid=Grid(cfg.Q)
    tracks=[]; next_id=1
    frozen_keys=set(); frozen_boxes=[]
    WIN="mementomori_live"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx=0; saved_idx=0
    try:
        while True:
            ok, frame=cap.read()
            if not ok:
                time.sleep(0.02); continue
            now=time.time()
            H,W=frame.shape[:2]

            if acc_fill is None or acc_lines is None or acc_fill.size!=(W,H) or acc_lines.size!=(W,H):
                acc_fill=Image.new("RGBA",(W,H),(0,0,0,0))
                acc_lines=Image.new("RGBA",(W,H),(0,0,0,0))
                grid.reset(W,H)
                tracks.clear(); next_id=1
                frozen_keys.clear(); frozen_boxes.clear()

            detections=[]
            if frame_idx % cfg.DETECT_EVERY==0:
                results=model(frame)
                persons=results.xyxy[0].cpu().numpy()
                detections=boxes_from_yolo_xyxy(persons,W,H,cfg.CONF)
                MAX_DETECTIONS = 8
                detections = sorted(detections, key=lambda b: b.area, reverse=True)[:MAX_DETECTIONS]

            # greedy match
            unmatched=set(range(len(detections)))
            live_idx=[i for i,t in enumerate(tracks) if not t.frozen]
            used=set()
            for di,det in enumerate(detections):
                best_i=-1; best_s=0.0
                for ti in live_idx:
                    if ti in used: continue
                    s=iou(det, tracks[ti].bbox)
                    if s>best_s: best_s=s; best_i=ti
                if best_i>=0 and best_s>=cfg.IOU_THRESH:
                    t=tracks[best_i]
                    tracks[best_i].bbox=lerp_box(t.bbox, det, cfg.SMOOTH_T)
                    tracks[best_i].last_seen_at=now
                    used.add(best_i)
                    unmatched.discard(di)

            for di in sorted(unmatched):
                det=detections[di]
                tracks.append(Track(next_id, det, now, now))
                next_id+=1

            tracks=[t for t in tracks if t.frozen or (now-t.last_seen_at)<=cfg.MISS_SECONDS]

            froze_any=False
            for t in tracks:
                if t.frozen: continue
                if (now-t.created_at)>=cfg.LIVE_SECONDS:
                    t.frozen=True
                    qb=quantize_box(clamp_box(t.bbox,W,H), cfg.Q)
                    k=geom_key(qb)
                    if qb.area>0 and k not in frozen_keys:
                        frozen_keys.add(k); frozen_boxes.append(qb)
                        stamp_frozen_outline(acc_lines, qb, cfg)
                        add_box_edges(grid, qb)
                        froze_any=True
            last_rect_time = 0
            if froze_any and (now - last_rect_time) > 0.6:
                last_rect_time = now
                if len(frozen_boxes) > 200:
                    rects = []
                else:
                    rects = find_closed_rectangles(grid)
                rects_sorted=sorted(set(rects), key=lambda b:b.area, reverse=True)
                rects_sorted = rects_sorted[:cfg.MAX_CELLS]
                #acc_fill.paste((0,0,0,0),(0,0,W,H))
                for b in frozen_boxes:
                    rng=random.Random(seed_from_rect("box",b))
                    stamp_fill(acc_fill,b,nature_imgs,cfg.FILL_ALPHA,rng)
                for b in rects_sorted:
                    rng=random.Random(seed_from_rect("cell",b))
                    stamp_fill(acc_fill,b,nature_imgs,cfg.FILL_ALPHA,rng)

            live_layer=Image.new("RGBA",(W,H),(0,0,0,0))
            draw_live_tracks(live_layer, tracks, cfg)
            out=Image.new("RGBA",(W,H),(0,0,0,0))
            out.alpha_composite(acc_fill); out.alpha_composite(acc_lines); out.alpha_composite(live_layer)

            cv2.imshow(WIN, pil_rgba_to_bgr(out))

            if froze_any:
                frame_path=output_dir/f"fixed_{saved_idx:06d}.png"
                out.save(frame_path)
                tmp=latest_path.with_suffix(".tmp.png")
                out.save(tmp); os.replace(tmp, latest_path)
                saved_idx+=1

            frame_idx+=1
            key=cv2.waitKey(1)&0xFF
            if key==ord("q") or key==27: break
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
