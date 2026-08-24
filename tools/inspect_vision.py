"""Inspect the shoggoth-mini vision pipeline visually.

Two modes:
  gt   — montage of dataset images with GROUND-TRUTH boxes (what the model is taught)
  pred — montage of images with a model's PREDICTED boxes + confidence (what it learned)

Examples (run from repo root, venv active):
  python tools/inspect_vision.py gt synthetic_dataset --split val --n 9
  python tools/inspect_vision.py pred runs/detect/yolo_training/exp30/weights/best.pt \
         synthetic_dataset/val/images --conf 0.25
"""
import argparse, glob, os, math
import cv2

TILE = 360  # px per cell


def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def montage(cells, out):
    n = len(cells)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    grid = None
    blank = None
    tiles = []
    for img in cells:
        t = cv2.resize(img, (TILE, TILE))
        tiles.append(t)
    while len(tiles) < rows * cols:
        tiles.append(tiles[0] * 0)  # black filler
    rowimgs = [cv2.hconcat(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = cv2.vconcat(rowimgs)
    cv2.imwrite(out, grid)
    print(f"wrote {out}  ({n} images, {rows}x{cols} grid)")


def mode_gt(args):
    img_dir = os.path.join(args.dataset, args.split, "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*")))[:args.n]
    cells = []
    for p in imgs:
        img = cv2.imread(p); h, w = img.shape[:2]
        lbl = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lbl):
            for line in open(lbl):
                c, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                draw_box(img, x1, y1, x2, y2, "tip (GT)", (0, 0, 255))  # red = truth
        cells.append(img)
    montage(cells, args.out)


def mode_pred(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    src = args.images
    imgs = sorted(glob.glob(os.path.join(src, "*"))) if os.path.isdir(src) else sorted(glob.glob(src))
    imgs = imgs[:args.n]
    res = model.predict(imgs, conf=args.conf, verbose=False)
    cells = []
    for r, p in zip(res, imgs):
        img = cv2.imread(p)
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0]); conf = float(b.conf[0])
            draw_box(img, x1, y1, x2, y2, f"tip {conf:.2f}", (0, 255, 0))  # green = prediction
        cells.append(img)
    montage(cells, args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gt"); g.add_argument("dataset"); g.add_argument("--split", default="val")
    g.add_argument("--n", type=int, default=9); g.add_argument("--out", default="inspect_gt.png")
    g.set_defaults(func=mode_gt)
    pr = sub.add_parser("pred"); pr.add_argument("model"); pr.add_argument("images")
    pr.add_argument("--conf", type=float, default=0.25); pr.add_argument("--n", type=int, default=9)
    pr.add_argument("--out", default="inspect_pred.png"); pr.set_defaults(func=mode_pred)
    args = ap.parse_args(); args.func(args)
