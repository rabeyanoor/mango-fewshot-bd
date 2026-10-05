"""Build capture groups from the image metadata and DINOv2 crop embeddings.

usage: .venv/bin/python tools/build_groups.py <meta.csv> <dinov2_feats.npy> <out_dir>

The embeddings are produced on Kaggle by run.build_groups (vit_small_patch14_dinov2,
fruit crops at 224 px). Grouping itself is CPU-only and deterministic.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import build_groups  # noqa: E402


def main(meta_path, feats_path, out_dir):
    meta = pd.read_csv(meta_path)
    feats = np.load(feats_path).astype(np.float32)
    group, drop, stats = build_groups(meta, feats)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"idx": meta.idx, "group": group, "drop": drop}).to_csv(
        os.path.join(out_dir, "groups.csv"), index=False)
    sizes = pd.Series(group).value_counts()
    stats.update({"n_groups": int(len(sizes)), "max_group": int(sizes.max()),
                  "mean_group": round(float(sizes.mean()), 3), "n_drop": int(drop.sum())})
    with open(os.path.join(out_dir, "groups_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(stats)


if __name__ == "__main__":
    main(*sys.argv[1:4])
