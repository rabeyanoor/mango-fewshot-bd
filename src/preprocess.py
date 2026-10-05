"""Image preprocessing shared by the local check and the Kaggle prep kernel.

Every raw image goes through the same steps so that no dataset gets its own
"signature" a model could shortcut on:
  1. trim_borders  - removes uniform padding strips (MangoImageBD pads many
                     photos to 504x1120 with flat white/blue bars).
  2. fruit_box     - finds the fruit as the largest saturated blob (mangoes are
                     green/yellow/orange; backgrounds are mostly white/grey).
  3. square crop   - square crop around the fruit with a margin, falling back
                     to a centre crop when no plausible blob is found.
  4. resize        - to SIZE x SIZE.
A 64-bit DCT perceptual hash is also computed for duplicate / near-duplicate
grouping (see grouping.py).
"""
import numpy as np
import cv2

SIZE = 288


def trim_borders(a, tol=12):
    """Drop edge rows/cols that are a single flat colour (padding)."""
    a16 = a.astype(np.int16)
    rdev = np.abs(a16 - np.median(a16, axis=1, keepdims=True)).max(axis=(1, 2))
    cdev = np.abs(a16 - np.median(a16, axis=0, keepdims=True)).max(axis=(0, 2))
    r = np.where(rdev > tol)[0]
    c = np.where(cdev > tol)[0]
    if len(r) < 32 or len(c) < 32:
        return a
    return a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]


def fruit_box(a):
    """Bounding box (x0, y0, x1, y1) of the largest saturated blob, or None."""
    h, w = a.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(a, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # mango hues: orange..green (OpenCV hue 5..95), reasonably saturated
    mask = ((s > 60) & (v > 40) & (hue >= 5) & (hue <= 95)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / float(mask.size)
        if frac < 0.02 or frac > 0.9:
            continue
        # a fruit is a compact blob; table edges / wooden planks are long
        # strips that run into the image border
        fill = area / float(bw * bh)
        sides = int(x <= 1) + int(y <= 1) + int(x + bw >= W - 1) + int(y + bh >= H - 1)
        elong = max(bw, bh) / float(min(bw, bh))
        # mango skin is strongly saturated; wood / clutter much less so
        sat = float(s[lab == i].mean()) / 255.0
        score = area * fill * sat ** 2 * (0.3 ** sides) / max(1.0, elong - 1.0)
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    if best is None:
        return None
    x, y, bw, bh = best
    return (x / scale, y / scale, (x + bw) / scale, (y + bh) / scale)


def square_crop(a, box, margin=0.12):
    h, w = a.shape[:2]
    if box is None:
        side = min(h, w)
        cx, cy = w / 2, h / 2
    else:
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
        side = min(max(side, 0.35 * min(h, w)), min(h, w))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    s = int(round(side))
    return a[y0:y0 + s, x0:x0 + s]


def phash(a):
    """64-bit DCT perceptual hash as a python int."""
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def process(a, size=SIZE):
    """RGB uint8 array -> (size x size RGB uint8, found_fruit, phash)."""
    a = trim_borders(a)
    box = fruit_box(a)
    out = cv2.resize(square_crop(a, box), (size, size), interpolation=cv2.INTER_AREA)
    return out, box is not None, phash(out)


def load_rgb(path, max_side=1600):
    """Read with EXIF orientation applied, downscaled early for speed."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))  # fast JPEG DCT-domain downscale
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.BILINEAR)
    return np.asarray(im)
