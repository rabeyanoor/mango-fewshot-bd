"""Build the unified MangoFS-BD image cache from the three public datasets.

Runs on Kaggle (internet on). Output in OUT_DIR:
  crop.npy   (N, S, S, 3) uint8   fruit-centred square crops
  full.npy   (N, S, S, 3) uint8   trimmed full frame, centre square crop
  meta.csv   one row per image (dataset, raw label, canonical cultivar,
             path, exif time/camera, perceptual hash, crop flag)
"""
import csv
import glob
import os
import re
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

from preprocess import SIZE, load_rgb, process, trim_borders, square_crop

MENDELEY = {
    "MangoImageBD": "https://data.mendeley.com/public-files/datasets/hp2cdckpdr/files/"
                    "28bb7356-9476-4b74-82b3-b445232e254c/file_downloaded",
    "Mangifera2012": "https://data.mendeley.com/public-files/datasets/w5jg84txj8/files/"
                     "c34ac544-aea9-451a-bce9-953d1ca645a9/file_downloaded",
}
D2_ROOT_CANDIDATES = glob.glob("/kaggle/input/**/MangoClassify-12", recursive=True)

# Raw folder name -> canonical cultivar.  Same cultivar under different
# spellings across datasets is merged so that a "novel" cultivar is novel
# in every dataset at once.
CANON = {
    "amrapali": "Amrapali",
    "ashshina classic": "AshshinaClassic",
    "ashshina zhinuk": "AshshinaZhinuk",
    "banana mango": "Banana", "banana": "Banana",
    "bari-4": "Bari4", "bari 4": "Bari4", "bari4": "Bari4",
    "bari-7": "Bari7", "bari 7": "Bari7",
    "bari-11": "Bari11",
    "fazli classic": "Fazli", "fazli": "Fazli", "fazlee": "Fazli",
    "fazli shurmai": "FazliShurmai",
    "gourmoti": "Gourmoti",
    "harivanga": "Harivanga", "haribhanga": "Harivanga",
    "himsagor": "Himsagar", "himsagar": "Himsagar",
    "katimon": "Katimon",
    "langra": "Langra",
    "kanchon langra": "KanchonLangra",
    "rupali": "Rupali",
    "shada": "Shada",
    "gobindobhog": "GobindoBhog",
    "gopalbhog": "GopalBhog",
    "khrishapat": "Khirsapat",
    "ranibhog": "RaniBhog",
    "sundari": "Sundari",
    "mollika": "Mollika",
    "nilambori": "Nilambori",
}


def canon(name):
    # Mangifera2012 folders carry their image count, e.g. "Amrapali-252"
    k = re.sub(r"-\d{3}$", "", name.strip()).lower().replace("_", " ")
    if k not in CANON:
        raise KeyError(f"unknown cultivar folder {name!r}")
    return CANON[k]


def fetch(name, url, work):
    z = os.path.join(work, name + ".zip")
    if not os.path.exists(z):
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", z, url], check=True)
    d = os.path.join(work, name)
    if not os.path.isdir(d):
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        os.remove(z)
    return d


def list_images(root):
    exts = (".jpg", ".jpeg", ".png")
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts) and not f.startswith("."):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def exif_info(path):
    from PIL import Image
    try:
        ex = Image.open(path).getexif()
        sub = ex.get_ifd(0x8769)
        t = sub.get(36867) or ex.get(306) or ""
        return str(t), str(ex.get(272) or "").strip()
    except Exception:
        return "", ""


def work_one(args):
    ds, label, path = args
    try:
        a = load_rgb(path)
    except Exception:  # corrupt file
        return None
    crop, found, h = process(a)
    full = cv2.resize(square_crop(trim_borders(a), None), (SIZE, SIZE),
                      interpolation=cv2.INTER_AREA)
    t, cam = exif_info(path)
    return dict(dataset=ds, raw_label=label, cultivar=canon(label),
                path=path, exif_time=t, camera=cam, phash=f"{h:016x}",
                fruit_found=int(found), h=a.shape[0], w=a.shape[1]), crop, full


def main(out_dir="/kaggle/working", work="/tmp/mango"):
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    d1 = fetch("MangoImageBD", MENDELEY["MangoImageBD"], work)
    for p in list_images(d1):
        jobs.append(("MangoImageBD", os.path.basename(os.path.dirname(p)), p))
    d3 = fetch("Mangifera2012", MENDELEY["Mangifera2012"], work)
    for p in list_images(d3):
        jobs.append(("Mangifera2012", os.path.basename(os.path.dirname(p)), p))
    if not D2_ROOT_CANDIDATES:
        sys.exit("MangoClassify-12 input not attached")
    for p in list_images(D2_ROOT_CANDIDATES[0]):
        jobs.append(("MangoClassify12", os.path.basename(os.path.dirname(p)), p))
    labels = sorted({(j[0], j[1]) for j in jobs})
    print(len(jobs), "images;", labels, flush=True)
    for ds, lab in labels:
        canon(lab)  # fail fast on unknown folder names

    rows, crops, fulls = [], [], []
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(work_one, jobs, chunksize=16)):
            if r is None:
                print("skip corrupt", jobs[i][2])
                continue
            m, c, f = r
            m["idx"] = len(rows)
            rows.append(m)
            crops.append(c)
            fulls.append(f)
            if i % 1000 == 0:
                print(i, flush=True)
    np.save(os.path.join(out_dir, "crop.npy"), np.stack(crops))
    np.save(os.path.join(out_dir, "full.npy"), np.stack(fulls))
    with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows), "fruit found:", sum(r["fruit_found"] for r in rows))
    # contact sheet per dataset for visual QA of the crops
    rng = np.random.default_rng(0)
    for ds in sorted({r["dataset"] for r in rows}):
        ids = [r["idx"] for r in rows if r["dataset"] == ds]
        pick = rng.choice(ids, size=min(60, len(ids)), replace=False)
        tiles = [cv2.resize(crops[i], (128, 128)) for i in pick]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        sheet = np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)])
        cv2.imwrite(os.path.join(out_dir, f"sheet_{ds}.jpg"), sheet[..., ::-1])


if __name__ == "__main__":
    main()
