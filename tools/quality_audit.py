"""Image-quality audit of the benchmark cache (resolution, blur, exposure, rotated or
mirrored copies).

usage: .venv/bin/python tools/quality_audit.py [benchmark_dir] [out_dir]

benchmark_dir holds crop.npy, meta.csv and groups.csv. Nothing is removed; the script
reports counts per source and lists the flagged images in <out_dir>/quality.json.

  low resolution  shorter side of the original photograph below 224 px
  blur            variance of the Laplacian of the 288 px grey crop below 100
  exposure        more than 20% of the crop pixels clipped (HSV value <= 5 or >= 250)
  dihedral copy   two images identical up to a 90-degree rotation or a mirror
                  (equal perceptual hash after the transform and mean absolute pixel
                  difference below 3 grey levels)
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocess import phash  # noqa: E402

BENCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "benchmark")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "verify")
os.makedirs(OUT, exist_ok=True)

meta = pd.read_csv(os.path.join(BENCH, "meta.csv"))
grp = pd.read_csv(os.path.join(BENCH, "groups.csv"))
meta = meta.merge(grp[["idx", "group", "drop"]], on="idx")
crop = np.load(os.path.join(BENCH, "crop.npy"), mmap_mode="r")


def dihedral(a):
    """The 8 rotations / mirrors of an image; index 0 is the identity."""
    out = []
    for f in (a, a[:, ::-1]):
        for k in range(4):
            out.append(np.rot90(f, k))
    return out


blur, clip, canon = [], [], []
for i in range(len(crop)):
    a = np.asarray(crop[i])
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    v = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 2]
    clip.append(float(((v <= 5) | (v >= 250)).mean()))
    canon.append([phash(np.ascontiguousarray(t)) for t in dihedral(a)])
meta["blur"] = blur
meta["clipped"] = clip
meta["low_res"] = (meta[["h", "w"]].min(axis=1) < 224).astype(int)

# rotated / mirrored copies: hash of a transformed image equals the hash of another image
h0 = {}
for i, hs in enumerate(canon):
    h0.setdefault(hs[0], []).append(i)
pairs = set()
for i, hs in enumerate(canon):
    for t, h in enumerate(hs[1:], start=1):
        for j in h0.get(h, []):
            if j == i:
                continue
            a = dihedral(np.asarray(crop[i]))[t].astype(np.int16)
            if np.abs(a - np.asarray(crop[j]).astype(np.int16)).mean() < 3:
                pairs.add((min(i, j), max(i, j), t))
rows = []
for i, j, t in sorted(pairs):
    rows.append({"a": int(i), "b": int(j), "transform": int(t),
                 "same_group": bool(meta.group[i] == meta.group[j]),
                 "same_cultivar": bool(meta.cultivar[i] == meta.cultivar[j]),
                 "a_path": meta.path[i], "b_path": meta.path[j]})

flags = {"low_res": meta.low_res == 1, "blur": meta.blur < 100, "exposure": meta.clipped > 0.2}
summary = {k: {d: int(v[meta.dataset == d].sum()) for d in sorted(meta.dataset.unique())}
           for k, v in flags.items()}
summary["dihedral_copies"] = {"pairs": len(rows),
                              "different_group": sum(not r["same_group"] for r in rows),
                              "different_cultivar": sum(not r["same_cultivar"] for r in rows)}
summary["blur_median"] = {d: round(float(meta.blur[meta.dataset == d].median()), 1)
                          for d in sorted(meta.dataset.unique())}
listing = {k: meta.loc[v, ["idx", "dataset", "cultivar", "path", "blur", "clipped", "h", "w"]]
           .to_dict("records") for k, v in flags.items()}
json.dump({"summary": summary, "flagged": listing, "dihedral_pairs": rows},
          open(os.path.join(OUT, "quality.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
