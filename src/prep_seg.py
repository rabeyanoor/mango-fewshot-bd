"""Background-removed variant of the image cache (seg.npy).

    import prep_seg; prep_seg.main()

The fruit crops of crop.npy are segmented with the U^2-Net salient-object model
(rembg, model "u2net") and composited on a uniform white background, so that the
background of the source dataset no longer differs between sources. The same
processing is applied to every image. If the mask covers less than 2% of the crop
(nothing found), the original crop is kept and flagged.

Writes to the working directory:
  seg.npy        uint8 (N, 288, 288, 3), aligned with meta.csv
  meta.csv       copy of the cache metadata (so the variant can be found)
  seg_stats.csv  idx, mask_frac (mean alpha), fallback (1 = original crop kept)
"""
import glob
import os
import shutil
import subprocess
import sys
import time

INPUT = os.environ.get("MANGOFS_INPUT", "/kaggle/input")
WORK = os.environ.get("MANGOFS_WORK", "/kaggle/working")


def composite(img, alpha):
    a = alpha.astype("float32")[..., None] / 255.0
    return (img.astype("float32") * a + 255.0 * (1.0 - a)).round().astype("uint8")


def main(model="u2net", min_frac=0.02, limit=None):
    try:
        import rembg  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rembg[cpu]"], check=True)
    import numpy as np
    import pandas as pd
    from PIL import Image
    from rembg import new_session, remove

    root = os.path.dirname(glob.glob(f"{INPUT}/**/crop.npy", recursive=True)[0])
    crop = np.load(os.path.join(root, "crop.npy"), mmap_mode="r")
    shutil.copy(os.path.join(root, "meta.csv"), os.path.join(WORK, "meta.csv"))
    os.environ.setdefault("U2NET_HOME", "/tmp/u2net")  # keep the model out of the outputs
    sess = new_session(model)
    n = len(crop) if limit is None else limit  # limit: quick test on the first images
    out = np.lib.format.open_memmap(os.path.join(WORK, "seg.npy"), mode="w+",
                                    dtype=np.uint8, shape=(n,) + crop.shape[1:])
    rows, t0 = [], time.time()
    for i in range(n):
        img = np.asarray(crop[i])
        alpha = np.asarray(remove(Image.fromarray(img), session=sess, only_mask=True))
        frac = float(alpha.mean() / 255.0)
        fallback = int(frac < min_frac)
        out[i] = img if fallback else composite(img, alpha)
        rows.append({"idx": i, "mask_frac": round(frac, 4), "fallback": fallback})
        if i % 500 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(WORK, "seg_stats.csv"), index=False)
    print("done:", len(s), "images, fallback", int(s.fallback.sum()),
          "median mask fraction", float(s.mask_frac.median()), flush=True)
