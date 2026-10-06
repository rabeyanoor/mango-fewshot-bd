"""Independent checks of the MangoFS-BD benchmark files (data/benchmark/).

usage: .venv/bin/python tools/verify_benchmark.py [benchmark_dir] [out_dir]

benchmark_dir must hold the image arrays (crop.npy, full.npy) besides meta.csv and groups.csv.

Writes <out_dir>/checks.json (one entry per check: name, passed, detail) and
figures that let a person inspect the processing by eye:
  samples.png        random crops of every cultivar from every source
  crop_steps.png     full frame vs fruit crop for random images of each source
  groups.png         members of random multi-image capture groups
  multisource.png    cultivars present in several sources, side by side per source
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import cultivar_folds  # noqa: E402

BENCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "benchmark")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "verify")
os.makedirs(OUT, exist_ok=True)
DSS = ["MangoImageBD", "MangoClassify12", "Mangifera2012"]
# image counts reported by the data papers / repositories (MangoImageBD: original photos only)
PUBLISHED = {"MangoImageBD": 5703, "MangoClassify12": 3900, "Mangifera2012": 2012}

meta = pd.read_csv(os.path.join(BENCH, "meta.csv"))
grp = pd.read_csv(os.path.join(BENCH, "groups.csv"))
m = meta.merge(grp, on="idx")
crop = np.load(os.path.join(BENCH, "crop.npy"), mmap_mode="r")
full = np.load(os.path.join(BENCH, "full.npy"), mmap_mode="r")
checks = []


def check(name, ok, detail):
    checks.append({"name": name, "passed": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + " - " + detail)


# 1. counts per source
cnt = meta.dataset.value_counts().to_dict()
check("image count per source equals the published count",
      all(cnt.get(d) == n for d, n in PUBLISHED.items()),
      ", ".join(f"{d}: {cnt.get(d)} (published {n})" for d, n in PUBLISHED.items()))

# 2. Mangifera2012 folder names carry the image count ("Amrapali-252")
mf = meta[meta.dataset == "Mangifera2012"].groupby("raw_label").size()
bad = {k: v for k, v in mf.items() if int(k.rsplit("-", 1)[1]) != v}
check("Mangifera2012 folder counts match the files read", not bad,
      "all 10 folders match" if not bad else f"mismatch: {bad}")

# 3. MangoImageBD: only original photographs (no processed / augmented copies)
mi = meta[meta.dataset == "MangoImageBD"].path
check("MangoImageBD uses only the original photographs",
      mi.str.contains("/MangoOriginal/").all() and not mi.str.contains("(?i)augment|process").any(),
      f"{len(mi)} paths, all under MangoOriginal/")

# 4. label harmonisation: each raw label maps to one cultivar
lab = meta.groupby(["dataset", "raw_label"]).cultivar.nunique()
check("each source folder maps to exactly one cultivar", (lab == 1).all(),
      f"{len(lab)} source folders -> {meta.cultivar.nunique()} cultivars")

# 5. cached images are valid (shape, not blank)
std = np.array([crop[i].std() for i in range(len(crop))])
check("every cached crop is a 288x288 RGB image with content",
      crop.shape[1:] == (288, 288, 3) and len(crop) == len(meta) and (std > 5).all(),
      f"{len(crop)} images, min pixel std {std.min():.1f}")

# 6. fruit found by the crop step
nf = meta[meta.fruit_found == 0]
check("fruit located in (almost) every image", len(nf) <= 0.001 * len(meta),
      f"{len(nf)} image(s) fell back to a centre crop: " + "; ".join(
          f"{r.dataset}/{r.cultivar}/{os.path.basename(r.path)}" for r in nf.itertuples()))

# 7. capture groups never mix cultivars or sources
kept = m[m["drop"] == 0]
mix_c = (kept.groupby("group").cultivar.nunique() > 1).sum()
mix_d = (kept.groupby("group").dataset.nunique() > 1).sum()
check("no capture group mixes cultivars or source datasets", mix_c == 0 and mix_d == 0,
      f"{kept.group.nunique()} groups; mixed-cultivar {mix_c}, mixed-source {mix_d}")
check("only the cross-label duplicate pair was dropped", int(m["drop"].sum()) == 2,
      f"{int(m['drop'].sum())} images dropped; benchmark keeps {len(kept)} images")

# 8. identical images under two cultivar names (exact pHash and identical pixels)
dup = kept.groupby("phash").cultivar.nunique()
cand = kept[kept.phash.isin(dup[dup > 1].index)]
same_px = 0
for _, g in cand.groupby("phash"):
    ids = g.idx.to_numpy()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if g.cultivar.iat[i] != g.cultivar.iat[j] and np.array_equal(crop[ids[i]], crop[ids[j]]):
                same_px += 1
check("no pixel-identical image appears under two cultivar names", same_px == 0,
      f"{len(cand)} images share a pHash across cultivars ("
      + "; ".join(f"{r.dataset}/{r.cultivar}/{os.path.basename(r.path)}" for r in cand.itertuples())
      + f"); pixel-identical cross-cultivar pairs: {same_px}")

# 9. folds: cultivar-disjoint, each cultivar novel exactly once
folds = cultivar_folds(kept)
flat_f = [c for f in folds for c in f]
check("3 cultivar-disjoint folds, every cultivar novel exactly once",
      len(flat_f) == len(set(flat_f)) == kept.cultivar.nunique() == 24 and [len(f) for f in folds] == [8, 8, 8],
      "; ".join(f"fold {k + 1}: " + ", ".join(sorted(f)) for k, f in enumerate(folds)))

json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)
cults = sorted(kept.cultivar.unique())
n = 3
fig, axes = plt.subplots(len(cults), 3 * n, figsize=(3 * n * 0.9, len(cults) * 0.95))
for i, c in enumerate(cults):
    for j, d in enumerate(DSS):
        sub = kept[(kept.cultivar == c) & (kept.dataset == d)]
        pick = rng.choice(sub.idx.to_numpy(), min(n, len(sub)), replace=False) if len(sub) else []
        for k in range(n):
            ax = axes[i, j * n + k]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k < len(pick):
                ax.imshow(crop[pick[k]])
            if i == 0 and k == 0:
                ax.set_title(d, fontsize=8, loc="left")
    axes[i, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=7)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "samples.png"), dpi=90)
plt.close(fig)

fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for r, d in enumerate(DSS):
    ids = rng.choice(kept[kept.dataset == d].idx.to_numpy(), 4, replace=False)
    for k, i in enumerate(ids):
        for col, (img, t) in enumerate(((full[i], "full frame"), (crop[i], "fruit crop"))):
            ax = axes[r, 2 * k + col]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{d[:12]} · {t}" if col == 0 else t, fontsize=6, loc="left")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "crop_steps.png"), dpi=100)
plt.close(fig)

multi = kept.groupby("group").filter(lambda g: len(g) >= 4)
gids = []
for d in DSS:
    gs = multi[multi.dataset == d].group.unique()
    gids += list(rng.choice(gs, min(2, len(gs)), replace=False))
fig, axes = plt.subplots(len(gids), 6, figsize=(9, 1.6 * len(gids)))
for r, gid in enumerate(gids):
    g = kept[kept.group == gid]
    for k in range(6):
        ax = axes[r, k]; ax.set_xticks([]); ax.set_yticks([])
        if k < len(g):
            ax.imshow(crop[g.idx.iat[k]])
    axes[r, 0].set_title(f"{g.dataset.iat[0]} · {g.cultivar.iat[0]} · group of {len(g)}",
                         fontsize=7, loc="left")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "groups.png"), dpi=100)
plt.close(fig)
multi_c = [c for c in cults if kept[kept.cultivar == c].dataset.nunique() > 1]
fig, axes = plt.subplots(len(multi_c), 12, figsize=(13, 1.15 * len(multi_c)))
for i, c in enumerate(multi_c):
    for j, d in enumerate(DSS):
        sub = kept[(kept.cultivar == c) & (kept.dataset == d)]
        pick = rng.choice(sub.idx.to_numpy(), min(4, len(sub)), replace=False) if len(sub) else []
        for k in range(4):
            ax = axes[i, 4 * j + k]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k < len(pick):
                ax.imshow(crop[pick[k]])
            if i == 0 and k == 0:
                ax.set_title(d, fontsize=8, loc="left")
    axes[i, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "multisource.png"), dpi=90)
plt.close(fig)
print("wrote", OUT)
