# MangoFS-BD: few-shot recognition of Bangladeshi mango cultivars

Given only 1, 5 or 10 labelled photos of a mango cultivar that a model has
never seen during training, how accurately can it recognise further photos of
that cultivar? This repository contains the benchmark, code, raw results and
run logs of a study that answers this question for Bangladeshi cultivars.

**Dataset download:**
[MangoFS-BD.zip](https://github.com/rabeyanoor/mango-fewshot-bd/releases/download/v1.0/MangoFS-BD.zip)
(about 500 MB, [release v1.0](https://github.com/rabeyanoor/mango-fewshot-bd/releases/tag/v1.0))

## Contents

1. [Overview](#overview)
2. [Main results](#main-results)
3. [Benchmark](#benchmark)
4. [Evaluation protocols](#evaluation-protocols)
5. [Methods](#methods)
6. [Repository layout](#repository-layout)
7. [Quick start](#quick-start)
8. [Re-running the experiments](#re-running-the-experiments)
9. [Licence and data](#licence-and-data)

## Overview

- **Benchmark.** Three public Bangladeshi mango datasets merged into one
  24-cultivar benchmark of 11,613 fruit images. Photos of the same fruit
  (multi-view shots, near-duplicates, re-encoded copies) are grouped, and a
  group is never split between training and test.
- **Controlled comparison.** Prototypical Networks, cross-entropy, ArcFace,
  supervised contrastive and triplet training on Conv-4, ResNet-18, ResNet-50,
  DenseNet-121 and DINOv2 ViT-S/14 and ViT-B/14. Every model is scored on the
  same test episodes, with paired significance tests.
- **Kolmogorov–Arnold networks (KANs).** A KAN projection head as an
  alternative to the MLP head, and a learnable KAN distance for Prototypical
  Networks.
- **Follow-up experiments.** Background removal, lightweight CNNs
  (EfficientNet-B0, MobileNetV2) and feature fusion of several backbones.
- **Ordinary fine-tuning versus few-shot inference** at matched numbers of
  labelled images.
- **All raw results and logs.** Every table can be regenerated from
  `results/` on a CPU.

## Main results

5-way accuracy on unseen cultivars (mean of three cultivar folds):

| Model | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| Conv-4 ProtoNet, trained from scratch | 64.1 | 74.2 | 77.3 |
| Best single model: DINOv2 ViT-B/14 trained with cross-entropy, backbone feature | 74.8 | 84.0 | 86.7 |
| Feature fusion of five cross-entropy-trained backbones | 76.1 | 85.0 | 87.6 |
| Feature fusion, logistic regression | 75.7 | 86.3 | 90.2 |

Findings:

- **Unseen cultivars are much harder than known ones.** Ordinary fine-tuning
  reaches 95.8–99.5% when every cultivar is known in advance. On unseen
  cultivars, the best model per pretrained backbone reaches 68–75% with one
  photo and 78–84% with five.
- **Projection heads hurt transfer.** An MLP or KAN head lowers accuracy on
  unseen cultivars. The best embedding is the backbone feature of a model that
  was trained with a head and is then used without it.
- **KAN does not help.** The KAN head is on par with the MLP head (−1.9 to
  +0.6 points) with about ten times as many parameters, and the KAN metric
  helps only DenseNet-121 (about +1 point). This is reported as a negative
  result.
- **A change of source dataset costs the most accuracy.** When support and
  query photos come from different source datasets (other phone, region,
  background, ripeness), 5-shot accuracy is only 46–68%, although these tasks
  have only the 2 or 3 cultivars both sources share (chance level 45%).
  Cultivars that appear in several source datasets are the hardest.
- **Feature fusion adds about one point; background removal does not help.**
  Fusing five backbones gives the best in-domain accuracy but not across
  source datasets. Removing the background with U²-Net lowers accuracy for all
  seven backbones tested. Without training, EfficientNet-B0 and MobileNetV2
  are as good as the ResNets.
- **A larger backbone adds little.** DINOv2 ViT-B/14 is at most one point
  more accurate than ViT-S/14, with four times the parameters.
- **The differences are not seed noise.** Over three training seeds, accuracy
  varies by a standard deviation of at most 1 point, while the gaps between
  heads are 3–5 points.
- **Leakage inflates closed-set accuracy by 0.9–2.6 points**, because
  closed-set accuracy is already close to the ceiling.
- **With 5 or more photos per known cultivar, ordinary fine-tuning beats
  prototypes.** With a single photo, prototypes are as good or better.

All tables are in [`results/tables.md`](results/tables.md).

<p align="center"><img src="figures/fig_main_accuracy.png" width="90%"></p>
<p align="center"><em>Accuracy on unseen cultivars (5-way, mean of three folds, 95% CI).</em></p>

More figures:
[confusion matrices](figures/fig_confusion.png),
[fine-tuning vs few-shot, DenseNet-121](figures/fig_finetune_vs_fewshot_densenet121.png),
[fine-tuning vs few-shot, ResNet-18](figures/fig_finetune_vs_fewshot_resnet18.png),
[learned KAN metric functions](figures/fig_kan_metric_paper.png),
[cultivars across source datasets](figures/fig_multisource.png).

## Benchmark

No new photos were taken. Three public datasets (CC BY 4.0) are merged:

| Source | Cultivars | Images | Notes |
|---|---|---|---|
| [MangoImageBD](https://data.mendeley.com/datasets/hp2cdckpdr/2) | 15 | 5,703 | original photos only (augmented copies excluded) |
| [MangoClassify-12](https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd) | 12 | 3,900 | 4 phone models, EXIF timestamps |
| [Mangifera2012](https://data.mendeley.com/datasets/w5jg84txj8) | 10 | 2,012 | iPhone 14 Pro Max, EXIF timestamps |
| **Merged** | **24** | **11,613** | cultivar names harmonised; one cross-label near-duplicate pair removed |

- **Preprocessing.** Padding bars are trimmed, a square is cropped around the
  fruit (the most fruit-like saturated blob) and resized to 288×288. A
  full-frame variant keeps the background for an ablation, and a third variant
  removes the background with U²-Net.
- **Capture groups.** Photos of the same fruit taken seconds apart
  (consecutive EXIF timestamps of one camera, at most 10 s apart),
  near-identical images (DINOv2 cosine similarity ≥ 0.97) and re-encoded
  copies (perceptual hash within 2 bits and cosine ≥ 0.97) form one group.
  Support and query sets never share a group, and closed-set test splits hold
  out whole groups. [`docs/grouping.md`](docs/grouping.md) explains how the
  thresholds were chosen.
- **Metadata** ([`benchmark/`](benchmark)). `meta.csv` lists every image with
  its source, original label, cultivar, camera and timestamp; `groups.csv`
  gives its capture group and whether it was dropped (`drop = 1`).
- **Checks.** `tools/verify_benchmark.py` confirms that the image counts equal
  the published counts, every source folder maps to one cultivar, no group
  mixes cultivars and the folds are cultivar-disjoint.
  `tools/quality_audit.py` reports blur, exposure and rotated or mirrored
  copies per source.

**The zip archive** contains every processed image as a 288×288 JPEG, as a
fruit crop (`images/crop/`) and as a full frame (`images/full/`), in one
folder per cultivar. It also contains `metadata.csv` (source, original label,
capture group, `drop` flag) and the exact splits of every protocol
(`splits/*.json`). It holds 11,615 images; the 2 images with `drop = 1` are
excluded from every split. The experiments read the same images from
uncompressed arrays, so JPEG compression causes small pixel differences.

## Evaluation protocols

| Protocol | What it measures |
|---|---|
| Unified | 3-fold cultivar cross-validation: 16 training and 8 unseen cultivars per fold; 5-way 1/5/10-shot tasks, 600 episodes each |
| Cross-domain | support photos from one source dataset, query photos of the same cultivars from another (2- or 3-way) |
| Open set | 5 known cultivars plus 1 unknown; AUROC for rejecting the unknown |
| Leave one dataset out | train on two source datasets, test on the third (seen and unseen cultivars separately) |
| Closed set | K ∈ {1, 5, 10, 20, all} photos per cultivar: ordinary fine-tuning vs prototypes |
| Leakage | random image split vs capture-group split |

Metrics: accuracy and macro-F1 with 95% confidence intervals, per-cultivar
recall, confusion matrices, calibration error, and Wilcoxon signed-rank tests
on identical episodes with Holm correction.

## Methods

| Name in the code | Description |
|---|---|
| `proto`, `proto_mlp`, `proto_kanhead` | Prototypical Network with no head, an MLP head or a KAN head |
| `kan_metric` | ProtoNet + MLP head with the KAN distance `d(q,c) = ‖q−c‖² + Σⱼ φⱼ(|qⱼ−cⱼ|)` |
| `proto_margin` | ProtoNet + MLP head with a prototype margin in the loss |
| `ours` (KAN-Proto) | KAN distance and prototype margin together |
| `ce`, `arcface`, `supcon`, `triplet`, `proto_tri` | cross-entropy (cosine classifier), ArcFace, supervised contrastive, triplet (random, semi-hard, batch-hard or distance-weighted mining), ProtoNet + triplet |
| `imagenet` | pretrained features without any training on mango photos |
| `fusion_imagenet`, `fusion_ce` | concatenated backbone features of five pretrained or cross-entropy-trained backbones |
| per-episode fine-tuning | a pretrained network fine-tuned on the support set of each test episode |

Backbones: Conv-4 (trained from scratch), ResNet-18, ResNet-50, DenseNet-121,
DINOv2 ViT-S/14 and ViT-B/14; EfficientNet-B0 and MobileNetV2 without
training.

Test-time rules: nearest prototype on the head output `z`, on the backbone
feature `f`, on `f` centred on the base-class mean (CL2N), on `f` centred per
source dataset (transductive), and logistic regression on `z`.

## Repository layout

```
notebooks/MangoFS_BD.ipynb   the complete library and pipeline as one Kaggle notebook
tools/
  build_groups.py            build capture groups from metadata and DINOv2 embeddings
  verify_benchmark.py        independent checks of the benchmark
  quality_audit.py           image-quality audit
analysis/
  aggregate.py               result files -> results/tables.md and figures
  kan_curves.py              plot the learned KAN distance functions of a checkpoint
benchmark/                   meta.csv (every image) and groups.csv (capture groups)
docs/grouping.md             how the capture-group rule and its thresholds were chosen
figures/                     result figures
results/
  kaggle/<suite>/            raw results of each experiment suite, one JSON line per model
  tables.md                  all result tables
logs/                        Kaggle run log of the data preparation and of every suite
tests/                       unit tests
```

The notebook holds all library code. Section 2 of the notebook writes it to
`src/` as these modules:

| Module | Purpose |
|---|---|
| `prep_data.py` | download the three datasets, harmonise labels, build the image cache |
| `prep_seg.py` | background-removed variant of the image cache (U²-Net) |
| `preprocess.py` | padding trim, fruit crop, perceptual hash |
| `data.py` | image cache, capture groups, folds and splits, samplers, augmentation |
| `models.py` | backbones, MLP and KAN heads, KAN distance, feature fusion |
| `losses.py` | training objectives |
| `train.py` | train one configuration and evaluate it |
| `evaluate.py` | episodic evaluation, open set, metrics |
| `finetune.py` | per-episode and closed-set fine-tuning |
| `efficiency.py` | parameters, GMACs and latency |
| `run.py` | experiment suites; resumable driver for several GPUs or a TPU VM |
| `export.py` | package the benchmark as a zip archive |

## Quick start

Everything below runs on a CPU.

```bash
git clone https://github.com/rabeyanoor/mango-fewshot-bd.git
cd mango-fewshot-bd
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# write the library modules from the notebook to src/
mkdir -p src && .venv/bin/python -c "import json; [open(c['source'][0].split()[1], 'w').write(''.join(c['source'][1:])) for c in json.load(open('notebooks/MangoFS_BD.ipynb'))['cells'] if c['source'] and c['source'][0].startswith('%%writefile')]"

.venv/bin/python -m pytest -q tests/test_units.py   # unit tests
.venv/bin/python analysis/aggregate.py              # results/kaggle -> results/tables.md and figures
```

Other scripts (`<benchmark_dir>` holds `crop.npy`, `full.npy`, `meta.csv`
and `groups.csv`, as written by step 1 below):

```bash
.venv/bin/python tools/verify_benchmark.py <benchmark_dir> <out_dir>   # benchmark checks and figures
.venv/bin/python tools/quality_audit.py <benchmark_dir> <out_dir>      # image-quality audit
.venv/bin/python analysis/kan_curves.py <ckpt.pt> ...                  # learned KAN functions
```

## Re-running the experiments

All training ran on free Kaggle notebooks (2× T4 GPU).

1. **Image cache.** In a Kaggle CPU notebook with internet on and the
   MangoClassify-12 dataset attached, run section 2 of the notebook and then
   `import prep_data; prep_data.main()`. It downloads the two Mendeley
   datasets and writes `crop.npy`, `full.npy` and `meta.csv`.
2. **Capture groups.** Upload `benchmark/groups.csv` as a Kaggle dataset, or
   rebuild it: the suite `groups` computes DINOv2 embeddings, and
   `tools/build_groups.py <meta.csv> <feats.npy> <out_dir>` builds the groups.
3. **Experiments.** Open `notebooks/MangoFS_BD.ipynb` on Kaggle, add the
   image cache and `groups.csv` as inputs, and turn on GPU T4 ×2 and internet.
   In section 5, set `SUITE`, uncomment `run.main(SUITE)` and run the notebook.

| Suite | What it runs | Extra inputs |
|---|---|---|
| `unified_main` | every method and backbone on the 3 cultivar folds | |
| `lodo_main` | leave one dataset out | |
| `unified_extra` | training objectives and triplet mining policies | |
| `closed_set` | fine-tuning vs prototypes, leakage test | output of `lodo_main` |
| `episode_ft` | per-episode fine-tuning | |
| `efficiency` | parameters and latency | |
| `ablations` | training seeds and full-frame images | |
| `improve` | backbone-feature and source-centred rules on saved models; DINOv2 ViT-B/14 | outputs of `unified_main` and `lodo_main` |
| `seg_zero` | background-removed images; EfficientNet-B0 and MobileNetV2 (CPU) | output of `prep_seg.main()` |
| `fusion` | feature fusion of five backbones (CPU) | outputs of `unified_main` and `improve` |

Each suite writes one JSON line per model to `results_<suite>_<device>.jsonl`.
Copy these files to `results/kaggle/<suite>/` and run `analysis/aggregate.py`.
Suites can be resumed: if a run hits the time limit, run it again with its
previous output as an extra input, and finished configurations are skipped.
Episode seeds depend only on the protocol, fold and number of shots, so every
model is scored on the same episodes. Model checkpoints are saved on Kaggle
and are not stored in this repository.

## Licence and data

Code: MIT licence (see [`LICENSE`](LICENSE)). The processed images in the
release are derived from MangoImageBD, MangoClassify-12 and Mangifera2012,
which are licensed CC BY 4.0, and are redistributed under the same licence.
Please cite the three source datasets when you use the benchmark.
