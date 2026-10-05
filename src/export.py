"""Package the unified benchmark as a self-contained zip (MangoFS-BD.zip).

Layout of the archive
  MangoFS-BD/
    README.md
    metadata.csv          one row per image: id, file paths, source dataset,
                          original label, canonical cultivar, camera, EXIF time,
                          capture group, drop flag
    splits/unified_fold{0,1,2}.json   base / novel cultivars + image ids
    splits/lodo_{dataset}.json        train / test ids, seen / novel cultivars
    splits/closed_{dataset}.json      capture-group 70/30 train / test ids
    images/crop/<cultivar>/<id>.jpg   fruit-centred 288x288 crops
    images/full/<cultivar>/<id>.jpg   trimmed full frame, 288x288
"""
import json
import os
import zipfile

import cv2
import numpy as np

from data import DATASETS, protocol_split
from finetune import group_holdout

SHORT = {"MangoImageBD": "MIBD", "MangoClassify12": "MC12", "Mangifera2012": "MF12"}

README = """# MangoFS-BD

Unified few-shot benchmark of {n_cult} Bangladeshi mango cultivars ({n_img} images),
assembled from three public datasets (all CC BY 4.0):

* MangoImageBD (Mendeley hp2cdckpdr v2): original images only
* MangoClassify-12 (Kaggle researchersajid/mangoclassify-12-native-mango-dataset-from-bd)
* Mangifera2012 (Mendeley w5jg84txj8)

Please cite the three source datasets.

File names are `<cultivar>_<source>_<index>.jpg`, where source is MIBD (MangoImageBD),
MC12 (MangoClassify-12) or MF12 (Mangifera2012). The folder name is the label too.
Images: `images/crop` holds fruit-centred square crops and `images/full` holds the
trimmed full frame. Both are 288x288 JPEG.
`metadata.csv`: `group` is the capture group (images that must never be split
across train/test or support/query). Rows with `drop == 1` are near-identical
images with conflicting labels and are excluded from every split.

Splits:
* `unified_fold{{k}}`: 3-fold cultivar cross-validation; every cultivar is novel in exactly one fold.
* `lodo_<dataset>`: leave one dataset out (seen and novel cultivars listed).
* `closed_<dataset>`: closed-set split, whole capture groups held out (about 30%).

Per-cultivar image counts (source dataset columns):

{table}
"""


def export(cache, out_path, quality=95):
    """cache: data.Cache with groups attached (in_memory=False is enough)."""
    meta = cache.meta.copy()
    meta["id"] = [f"{c}_{SHORT[d]}_{i:05d}" for c, d, i in zip(meta.cultivar, meta.dataset, meta.idx)]
    meta["crop_path"] = "images/crop/" + meta.cultivar + "/" + meta.id + ".jpg"
    meta["full_path"] = "images/full/" + meta.cultivar + "/" + meta.id + ".jpg"
    full = np.load(os.path.join(cache.root, "full.npy"), mmap_mode="r")
    keep = ["id", "idx", "crop_path", "full_path", "dataset", "raw_label", "cultivar", "camera",
            "exif_time", "phash", "group", "drop"]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs are already compressed
        root = "MangoFS-BD/"
        z.writestr(root + "metadata.csv", meta[keep].to_csv(index=False))
        for f in range(3):
            tr, te, info = protocol_split(cache.meta, "unified", f)
            z.writestr(root + f"splits/unified_fold{f}.json",
                       json.dumps({**info, "train_idx": tr.tolist(), "test_idx": te.tolist()}))
        for f, ds in enumerate(DATASETS):
            tr, te, info = protocol_split(cache.meta, "lodo", f)
            z.writestr(root + f"splits/lodo_{ds}.json",
                       json.dumps({**info, "train_idx": tr.tolist(), "test_idx": te.tolist()}))
            m = cache.meta[(cache.meta.dataset == ds) & (cache.meta["drop"] == 0)]
            tr, te = group_holdout(m, seed=0)
            z.writestr(root + f"splits/closed_{ds}.json",
                       json.dumps({"train_idx": tr.tolist(), "test_idx": te.tolist()}))
        table = meta.pivot_table(index="cultivar", columns="dataset", values="idx",
                                 aggfunc="count", fill_value=0).to_markdown()
        z.writestr(root + "README.md", README.format(n_cult=meta.cultivar.nunique(),
                                                     n_img=len(meta), table=table))
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        for i, r in enumerate(meta.itertuples()):
            for arr, path in ((cache.images[r.idx], r.crop_path), (full[r.idx], r.full_path)):
                _, buf = cv2.imencode(".jpg", np.ascontiguousarray(arr[..., ::-1]), enc)
                z.writestr(root + path, buf.tobytes())
            if i % 2000 == 0:
                print("exported", i, flush=True)
    print("wrote", out_path, os.path.getsize(out_path) / 2 ** 20, "MB")
