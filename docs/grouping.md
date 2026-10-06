# Capture groups (leakage control)

Goal: two photographs of the same fruit, or near-copies of
one photograph, must never end up on opposite sides of a split, and never as
support and query of the same episode.

## Evidence used to choose the rule

All numbers are on the 11,615 unified images with DINOv2 ViT-S/14 embeddings
of the fruit crops (computed on Kaggle by `run.build_groups`).

**pHash alone does not work here.** Centred fruit on plain backgrounds gives
many low-Hamming pairs across cultivars:

| pHash Hamming ≤ | pairs | of which different cultivar |
|---|---|---|
| 0 | 27 | 2 |
| 2 | 209 | 35 |
| 4 | 1,318 | 513 |
| 6 | 6,574 | 3,610 |

Cross-cultivar pairs at Hamming ≤ 6 have a median DINOv2 cosine of 0.75, so
they are not duplicates. A first version of the grouping (pHash ≤ 6 plus single
linkage) dropped 2,487 images as "label conflicts" and built a 895-image group.
That version was discarded.

**Time vs. similarity** (consecutive shots of one cultivar and camera):

| gap | MangoClassify-12 median cos | Mangifera2012 median cos |
|---|---|---|
| ≤ 2 s | 0.988 | 0.872 |
| 2–5 s | 0.903 | 0.867 |
| 5–10 s | 0.883 | 0.856 |
| 10–20 s | 0.857 | 0.847 |
| > 60 s | 0.819 | 0.821 |

Different views of one fruit cannot be told apart from different fruits of the
same cultivar by appearance alone. Fruit identity is therefore approximated by
the capture sequence: all views of the same fruit or capture sequence are kept
together.

**Threshold grid** (groups per dataset, max / 99th-percentile group size):

| seq gap | dup cos | MangoImageBD | MangoClassify-12 | Mangifera2012 |
|---|---|---|---|---|
| 3 s | 0.95 | 3898, 6/4 | 3223, 7/4 | 1269, 11/7 |
| 5 s | 0.97 | 5040, 4/2 | 2745, 10/5 | 816, 16/11 |
| **10 s** | **0.97** | **5040, 4/2** | **1431, 35/12** | **365, 42/16** |
| 10 s | 0.95 | 3898, 6/4 | 1278, 64/17 | 312, 97/41 |

## Final rule (`src/data.py: build_groups` defaults)

* capture sequence: same dataset, cultivar and camera, consecutive EXIF
  timestamps ≤ 10 s apart;
* near-duplicates: complete-linkage clusters with every pairwise cosine ≥ 0.97
  (complete linkage avoids chaining among similar fruits of one cultivar);
* re-encoded copies: pHash Hamming ≤ 2 **and** cosine ≥ 0.97;
* label conflict: cosine ≥ 0.97 across different cultivars → both images
  dropped (1 pair found).

MangoImageBD has no EXIF data, so only near-duplicate grouping applies to it.
This limitation is stated in the paper. With this rule all 8 novel
cultivars of every `unified` fold can supply 10 support images plus 15
group-disjoint queries.
