# MangoFS-BD – results

_448 result records from `results/kaggle`._


## Table 0 – The unified MangoFS-BD benchmark (images / capture groups)

| cultivar        | Mangifera2012   | MangoClassify12   | MangoImageBD   |
|:----------------|:----------------|:------------------|:---------------|
| Amrapali        | 252 / 43        | 599 / 175         | 135 / 131      |
| AshshinaClassic | 0 / 0           | 0 / 0             | 571 / 491      |
| AshshinaZhinuk  | 0 / 0           | 0 / 0             | 1286 / 1061    |
| Banana          | 0 / 0           | 212 / 63          | 83 / 80        |
| Bari11          | 0 / 0           | 0 / 0             | 1244 / 1021    |
| Bari4           | 235 / 45        | 240 / 51          | 74 / 71        |
| Bari7           | 176 / 40        | 0 / 0             | 0 / 0          |
| Fazli           | 156 / 35        | 120 / 47          | 171 / 165      |
| FazliShurmai    | 0 / 0           | 0 / 0             | 247 / 243      |
| GobindoBhog     | 0 / 0           | 41 / 15           | 0 / 0          |
| GopalBhog       | 0 / 0           | 406 / 290         | 0 / 0          |
| Gourmoti        | 0 / 0           | 0 / 0             | 630 / 588      |
| Harivanga       | 202 / 21        | 575 / 147         | 265 / 261      |
| Himsagar        | 0 / 0           | 502 / 185         | 106 / 105      |
| KanchonLangra   | 210 / 46        | 0 / 0             | 0 / 0          |
| Katimon         | 163 / 38        | 0 / 0             | 424 / 358      |
| Khirsapat       | 0 / 0           | 379 / 232         | 0 / 0          |
| Langra          | 202 / 31        | 506 / 119         | 120 / 119      |
| Mollika         | 221 / 52        | 0 / 0             | 0 / 0          |
| Nilambori       | 195 / 14        | 0 / 0             | 0 / 0          |
| RaniBhog        | 0 / 0           | 92 / 63           | 0 / 0          |
| Rupali          | 0 / 0           | 0 / 0             | 184 / 184      |
| Shada           | 0 / 0           | 0 / 0             | 163 / 162      |
| Sundari         | 0 / 0           | 226 / 43          | 0 / 0          |

## Table 1 – Novel-cultivar 5-way accuracy (%), unified benchmark, rule `proto_z`

Mean over 3 cultivar folds × 600 episodes, ± 95% CI. Bold = best per backbone and shot. † = significantly different from *ours* (Wilcoxon, Holm, p<0.05).

| Method | Conv-4 1-shot | Conv-4 5-shot | Conv-4 10-shot | ResNet-18 1-shot | ResNet-18 5-shot | ResNet-18 10-shot | ResNet-50 1-shot | ResNet-50 5-shot | ResNet-50 10-shot | DenseNet-121 1-shot | DenseNet-121 5-shot | DenseNet-121 10-shot | DINOv2 ViT-S/14 1-shot | DINOv2 ViT-S/14 5-shot | DINOv2 ViT-S/14 10-shot | DINOv2 ViT-B/14 1-shot | DINOv2 ViT-B/14 5-shot | DINOv2 ViT-B/14 10-shot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet features (no training) | – | – | – | 63.33 ± 0.47 † | 75.33 ± 0.42 † | 79.05 ± 0.38 † | 63.50 ± 0.47 † | 75.00 ± 0.43 † | 78.26 ± 0.39 † | 62.11 ± 0.46 † | 74.74 ± 0.42 † | 78.50 ± 0.38 | 63.86 ± 0.46 | 79.43 ± 0.38 † | **83.93 ± 0.32 †** | 64.36 ± 0.45 | 79.06 ± 0.39 | 83.22 ± 0.34 |
| Fine-tuned CE (cosine) | **65.80 ± 0.50 †** | **75.51 ± 0.42 †** | **78.55 ± 0.39 †** | **67.59 ± 0.47 †** | **77.63 ± 0.39 †** | **80.27 ± 0.37 †** | 62.70 ± 0.47 | 74.84 ± 0.39 † | 78.26 ± 0.36 † | 67.94 ± 0.45 † | 78.06 ± 0.39 † | 81.05 ± 0.37 † | 67.87 ± 0.48 † | 77.85 ± 0.41 † | 81.09 ± 0.36 † | 66.64 ± 0.49 | 76.70 ± 0.42 | 79.71 ± 0.38 |
| ProtoNet | 64.09 ± 0.49 † | 74.19 ± 0.44 † | 77.32 ± 0.40 † | 66.09 ± 0.47 † | 76.62 ± 0.40 † | 79.49 ± 0.38 † | **67.82 ± 0.48 †** | **78.27 ± 0.39 †** | **81.51 ± 0.36 †** | **68.99 ± 0.47 †** | **79.14 ± 0.39 †** | **81.79 ± 0.36 †** | **69.61 ± 0.47 †** | **79.82 ± 0.39 †** | 82.84 ± 0.36 † | **70.20 ± 0.47** | **80.47 ± 0.39** | **83.61 ± 0.35** |
| ProtoNet + MLP head | 62.53 ± 0.50 † | 72.24 ± 0.44 † | 74.91 ± 0.41 † | 63.78 ± 0.50 † | 74.34 ± 0.42 † | 77.48 ± 0.40 † | 63.59 ± 0.48 † | 74.58 ± 0.40 † | 77.71 ± 0.37 † | 64.81 ± 0.48 | 75.17 ± 0.40 † | 78.01 ± 0.38 † | 65.56 ± 0.49 † | 75.25 ± 0.41 † | 78.21 ± 0.38 † | 65.36 ± 0.50 | 74.75 ± 0.44 | 77.82 ± 0.39 |
| ProtoNet + KAN head | 63.25 ± 0.50 † | 72.87 ± 0.44 † | 75.56 ± 0.40 † | 63.39 ± 0.49 † | 73.70 ± 0.42 † | 76.68 ± 0.40 † | 62.09 ± 0.47 † | 72.68 ± 0.42 † | 76.15 ± 0.38 † | 64.79 ± 0.48 | 75.47 ± 0.40 | 78.32 ± 0.38 † | 65.18 ± 0.50 † | 75.04 ± 0.42 | 77.92 ± 0.39 | – | – | – |
| + prototype margin | 62.71 ± 0.50 † | 72.11 ± 0.45 † | 74.84 ± 0.41 † | 63.20 ± 0.49 † | 73.58 ± 0.43 † | 76.68 ± 0.38 † | 61.31 ± 0.48 † | 72.25 ± 0.43 † | 75.54 ± 0.39 † | 64.42 ± 0.48 | 74.86 ± 0.41 † | 77.69 ± 0.38 † | 63.83 ± 0.48 † | 74.20 ± 0.42 † | 77.20 ± 0.39 † | – | – | – |
| + KAN metric | 62.14 ± 0.51 † | 70.66 ± 0.46 | 72.66 ± 0.43 † | 63.73 ± 0.50 † | 73.98 ± 0.43 † | 76.88 ± 0.41 † | 63.44 ± 0.50 † | 73.93 ± 0.42 | 77.01 ± 0.39 | 65.45 ± 0.47 † | 76.22 ± 0.39 † | 79.08 ± 0.37 † | 64.16 ± 0.48 † | 74.37 ± 0.41 † | 77.47 ± 0.38 † | – | – | – |
| KAN metric + margin (ours) | 61.64 ± 0.51 | 70.82 ± 0.45 | 73.25 ± 0.42 | 62.07 ± 0.49 | 72.87 ± 0.42 | 75.80 ± 0.38 | 62.66 ± 0.49 | 73.88 ± 0.41 | 77.11 ± 0.39 | 64.31 ± 0.48 | 75.63 ± 0.40 | 78.63 ± 0.37 | 64.47 ± 0.49 | 74.85 ± 0.42 | 77.82 ± 0.36 | – | – | – |
| ArcFace | – | – | – | 63.64 ± 0.50 † | 73.88 ± 0.43 † | 76.50 ± 0.39 † | – | – | – | 60.69 ± 0.49 † | 71.10 ± 0.42 † | 74.59 ± 0.39 † | – | – | – | – | – | – |
| SupCon | – | – | – | 59.65 ± 0.50 † | 69.67 ± 0.44 † | 72.13 ± 0.40 † | – | – | – | 54.71 ± 0.49 † | 65.29 ± 0.44 † | 68.83 ± 0.40 † | – | – | – | – | – | – |
| Triplet (batch-hard) | – | – | – | 60.33 ± 0.50 † | 70.78 ± 0.43 † | 73.87 ± 0.40 † | – | – | – | 62.08 ± 0.48 † | 72.63 ± 0.42 † | 75.43 ± 0.38 † | – | – | – | – | – | – |
| ProtoNet + triplet | – | – | – | 61.47 ± 0.49 † | 72.02 ± 0.43 † | 74.93 ± 0.39 † | – | – | – | 63.05 ± 0.49 † | 73.59 ± 0.42 † | 76.60 ± 0.38 † | – | – | – | – | – | – |
| Ours + KAN head | – | – | – | 62.16 ± 0.49 | 72.60 ± 0.42 | 75.55 ± 0.39 † | – | – | – | 63.22 ± 0.47 † | 74.43 ± 0.40 † | 77.39 ± 0.37 † | – | – | – | – | – | – |

## Table 2 – Inference rule on the same embeddings (5-shot, %)

`proto_z`: nearest prototype with the model's metric on the head output; `proto_f`: Euclidean prototype on backbone features; `proto_fc`: the same after base-mean centring (CL2N); `logreg_z`: logistic regression fitted on the support set.

| Backbone | Method | proto_z | proto_f | proto_fc | logreg_z |
|---|---|---|---|---|---|
| Conv-4 | Fine-tuned CE (cosine) | 75.51 ± 0.42 | 72.61 ± 0.42 | 72.72 ± 0.42 | 76.79 ± 0.42 |
| Conv-4 | ProtoNet | 74.19 ± 0.44 | 74.19 ± 0.44 | 73.99 ± 0.43 | 75.41 ± 0.42 |
| Conv-4 | ProtoNet + MLP head | 72.24 ± 0.44 | 72.93 ± 0.42 | 73.04 ± 0.42 | 73.39 ± 0.44 |
| Conv-4 | ProtoNet + KAN head | 72.87 ± 0.44 | 72.90 ± 0.42 | 72.97 ± 0.42 | 73.98 ± 0.43 |
| Conv-4 | + prototype margin | 72.11 ± 0.45 | 73.14 ± 0.42 | 73.23 ± 0.42 | 73.30 ± 0.44 |
| Conv-4 | + KAN metric | 70.66 ± 0.46 | 73.00 ± 0.42 | 72.96 ± 0.42 | 75.39 ± 0.43 |
| Conv-4 | KAN metric + margin (ours) | 70.82 ± 0.45 | 72.74 ± 0.43 | 72.87 ± 0.42 | 74.42 ± 0.44 |
| ResNet-18 | ImageNet features (no training) | 75.33 ± 0.42 | 75.33 ± 0.42 | 75.69 ± 0.42 | 77.54 ± 0.41 |
| ResNet-18 | Fine-tuned CE (cosine) | 77.63 ± 0.39 | 80.88 ± 0.37 | 81.07 ± 0.37 | 78.70 ± 0.39 |
| ResNet-18 | ProtoNet | 76.62 ± 0.40 | 76.62 ± 0.40 | 76.69 ± 0.40 | 78.87 ± 0.39 |
| ResNet-18 | ProtoNet + MLP head | 74.34 ± 0.42 | 80.70 ± 0.38 | 80.86 ± 0.37 | 75.32 ± 0.42 |
| ResNet-18 | ProtoNet + KAN head | 73.70 ± 0.42 | 80.97 ± 0.38 | 81.13 ± 0.38 | 75.11 ± 0.41 |
| ResNet-18 | + prototype margin | 73.58 ± 0.43 | 80.59 ± 0.38 | 80.79 ± 0.37 | 74.79 ± 0.42 |
| ResNet-18 | + KAN metric | 73.98 ± 0.43 | 80.33 ± 0.38 | 80.46 ± 0.38 | 75.25 ± 0.43 |
| ResNet-18 | KAN metric + margin (ours) | 72.87 ± 0.42 | 80.04 ± 0.38 | 80.19 ± 0.38 | 74.29 ± 0.41 |
| ResNet-18 | ArcFace | 73.88 ± 0.43 | 81.35 ± 0.38 | 81.50 ± 0.37 | 75.37 ± 0.41 |
| ResNet-18 | SupCon | 69.67 ± 0.44 | 81.24 ± 0.38 | 81.43 ± 0.38 | 71.53 ± 0.44 |
| ResNet-18 | Triplet (batch-hard) | 70.78 ± 0.43 | 81.14 ± 0.37 | 81.34 ± 0.37 | 72.91 ± 0.41 |
| ResNet-18 | ProtoNet + triplet | 72.02 ± 0.43 | 80.93 ± 0.38 | 81.08 ± 0.37 | 73.79 ± 0.41 |
| ResNet-18 | Ours + KAN head | 72.60 ± 0.42 | 79.48 ± 0.39 | 79.77 ± 0.38 | 74.17 ± 0.42 |
| ResNet-50 | ImageNet features (no training) | 75.00 ± 0.43 | 75.00 ± 0.43 | 75.39 ± 0.41 | 78.23 ± 0.40 |
| ResNet-50 | Fine-tuned CE (cosine) | 74.84 ± 0.39 | 81.10 ± 0.38 | 81.22 ± 0.38 | 76.89 ± 0.39 |
| ResNet-50 | ProtoNet | 78.27 ± 0.39 | 78.27 ± 0.39 | 78.38 ± 0.39 | 80.56 ± 0.38 |
| ResNet-50 | ProtoNet + MLP head | 74.58 ± 0.40 | 81.18 ± 0.39 | 81.30 ± 0.38 | 75.72 ± 0.39 |
| ResNet-50 | ProtoNet + KAN head | 72.68 ± 0.42 | 81.29 ± 0.38 | 81.40 ± 0.38 | 74.33 ± 0.41 |
| ResNet-50 | + prototype margin | 72.25 ± 0.43 | 81.48 ± 0.38 | 81.56 ± 0.38 | 73.76 ± 0.41 |
| ResNet-50 | + KAN metric | 73.93 ± 0.42 | 81.55 ± 0.38 | 81.63 ± 0.38 | 75.62 ± 0.41 |
| ResNet-50 | KAN metric + margin (ours) | 73.88 ± 0.41 | 81.56 ± 0.38 | 81.63 ± 0.38 | 75.69 ± 0.40 |
| DenseNet-121 | ImageNet features (no training) | 74.74 ± 0.42 | 74.74 ± 0.42 | 75.09 ± 0.42 | 77.91 ± 0.40 |
| DenseNet-121 | Fine-tuned CE (cosine) | 78.06 ± 0.39 | 82.60 ± 0.37 | 82.85 ± 0.36 | 79.23 ± 0.39 |
| DenseNet-121 | ProtoNet | 79.14 ± 0.39 | 79.14 ± 0.39 | 79.30 ± 0.39 | 81.40 ± 0.38 |
| DenseNet-121 | ProtoNet + MLP head | 75.17 ± 0.40 | 82.16 ± 0.37 | 82.29 ± 0.36 | 75.89 ± 0.41 |
| DenseNet-121 | ProtoNet + KAN head | 75.47 ± 0.40 | 81.81 ± 0.37 | 82.08 ± 0.37 | 75.70 ± 0.40 |
| DenseNet-121 | + prototype margin | 74.86 ± 0.41 | 81.78 ± 0.37 | 81.89 ± 0.37 | 75.67 ± 0.40 |
| DenseNet-121 | + KAN metric | 76.22 ± 0.39 | 81.77 ± 0.37 | 81.92 ± 0.37 | 76.84 ± 0.40 |
| DenseNet-121 | KAN metric + margin (ours) | 75.63 ± 0.40 | 81.85 ± 0.37 | 82.12 ± 0.37 | 76.60 ± 0.40 |
| DenseNet-121 | ArcFace | 71.10 ± 0.42 | 82.87 ± 0.36 | 83.02 ± 0.36 | 74.34 ± 0.40 |
| DenseNet-121 | SupCon | 65.29 ± 0.44 | 82.70 ± 0.37 | 82.87 ± 0.36 | 68.78 ± 0.43 |
| DenseNet-121 | Triplet (batch-hard) | 72.63 ± 0.42 | 82.67 ± 0.37 | 82.89 ± 0.37 | 74.39 ± 0.40 |
| DenseNet-121 | ProtoNet + triplet | 73.59 ± 0.42 | 82.61 ± 0.36 | 82.87 ± 0.36 | 75.31 ± 0.41 |
| DenseNet-121 | Ours + KAN head | 74.43 ± 0.40 | 81.76 ± 0.38 | 81.89 ± 0.38 | 75.66 ± 0.41 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | 79.43 ± 0.38 | 79.43 ± 0.38 | 80.01 ± 0.36 | 81.75 ± 0.36 |
| DINOv2 ViT-S/14 | Fine-tuned CE (cosine) | 77.85 ± 0.41 | 83.29 ± 0.35 | 83.45 ± 0.35 | 79.72 ± 0.39 |
| DINOv2 ViT-S/14 | ProtoNet | 79.82 ± 0.39 | 79.82 ± 0.39 | 79.79 ± 0.38 | 81.83 ± 0.37 |
| DINOv2 ViT-S/14 | ProtoNet + MLP head | 75.25 ± 0.41 | 82.09 ± 0.37 | 82.13 ± 0.37 | 76.69 ± 0.40 |
| DINOv2 ViT-S/14 | ProtoNet + KAN head | 75.04 ± 0.42 | 82.12 ± 0.37 | 82.21 ± 0.37 | 76.44 ± 0.41 |
| DINOv2 ViT-S/14 | + prototype margin | 74.20 ± 0.42 | 82.73 ± 0.37 | 82.79 ± 0.36 | 75.87 ± 0.41 |
| DINOv2 ViT-S/14 | + KAN metric | 74.37 ± 0.41 | 80.69 ± 0.38 | 80.83 ± 0.37 | 75.92 ± 0.41 |
| DINOv2 ViT-S/14 | KAN metric + margin (ours) | 74.85 ± 0.42 | 82.37 ± 0.36 | 82.44 ± 0.36 | 76.91 ± 0.41 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | 79.06 ± 0.39 | 79.06 ± 0.39 | 79.60 ± 0.38 | 82.17 ± 0.37 |
| DINOv2 ViT-B/14 | Fine-tuned CE (cosine) | 76.70 ± 0.42 | 83.99 ± 0.37 | 84.12 ± 0.37 | 79.24 ± 0.40 |
| DINOv2 ViT-B/14 | ProtoNet | 80.47 ± 0.39 | 80.47 ± 0.39 | 80.55 ± 0.39 | 82.75 ± 0.37 |
| DINOv2 ViT-B/14 | ProtoNet + MLP head | 74.75 ± 0.44 | 82.76 ± 0.37 | 82.78 ± 0.37 | 76.86 ± 0.42 |

## Table 4 – Leave-one-dataset-out, 5-way 5-shot accuracy (%)

Train on two datasets, test on every cultivar of the third. *seen* = cultivar also present in training (pure domain shift); *novel* = unseen cultivar and unseen domain.

| Backbone | Method | MangoImageBD all | MangoImageBD seen | MangoImageBD novel | MangoClassify12 all | MangoClassify12 seen | MangoClassify12 novel | Mangifera2012 all | Mangifera2012 seen | Mangifera2012 novel |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | ImageNet features (no training) | 79.46 ± 0.73 | 82.72 ± 0.58 | 79.14 ± 0.59 | 68.80 ± 0.83 | 60.15 ± 0.72 | 77.83 ± 0.51 | 74.73 ± 0.80 | 76.81 ± 0.59 | 71.34 ± 0.58 |
| ResNet-18 | Fine-tuned CE (cosine) | 78.97 ± 0.67 | 83.12 ± 0.50 | 75.80 ± 0.57 | 66.58 ± 0.73 | 60.70 ± 0.71 | 71.37 ± 0.53 | 76.74 ± 0.71 | 80.80 ± 0.51 | 71.03 ± 0.53 |
| ResNet-18 | ProtoNet + MLP head | 75.72 ± 0.68 | 82.65 ± 0.52 | 70.46 ± 0.58 | 60.31 ± 0.75 | 55.61 ± 0.73 | 65.82 ± 0.56 | 68.60 ± 0.74 | 75.50 ± 0.60 | 56.43 ± 0.65 |
| ResNet-18 | KAN metric + margin (ours) | 74.59 ± 0.70 | 82.58 ± 0.52 | 65.99 ± 0.59 | 60.55 ± 0.76 | 55.28 ± 0.69 | 64.77 ± 0.57 | 73.30 ± 0.76 | 77.78 ± 0.57 | 65.60 ± 0.60 |
| ResNet-50 | ImageNet features (no training) | 78.71 ± 0.75 | 80.67 ± 0.60 | 78.24 ± 0.52 | 67.01 ± 0.86 | 60.02 ± 0.79 | 71.99 ± 0.56 | 76.26 ± 0.72 | 75.12 ± 0.64 | 72.30 ± 0.58 |
| ResNet-50 | Fine-tuned CE (cosine) | 78.30 ± 0.65 | 81.86 ± 0.54 | 75.75 ± 0.55 | 66.61 ± 0.70 | 60.54 ± 0.67 | 71.73 ± 0.51 | 73.58 ± 0.75 | 73.68 ± 0.61 | 73.63 ± 0.57 |
| ResNet-50 | ProtoNet + MLP head | 75.98 ± 0.66 | 81.04 ± 0.55 | 71.87 ± 0.56 | 59.28 ± 0.75 | 51.92 ± 0.66 | 66.85 ± 0.57 | 70.12 ± 0.74 | 71.76 ± 0.65 | 66.99 ± 0.59 |
| ResNet-50 | KAN metric + margin (ours) | 73.18 ± 0.71 | 78.02 ± 0.56 | 65.86 ± 0.61 | 60.44 ± 0.72 | 56.87 ± 0.70 | 64.41 ± 0.58 | 69.57 ± 0.78 | 75.96 ± 0.57 | 61.23 ± 0.58 |
| DenseNet-121 | ImageNet features (no training) | 79.86 ± 0.70 | 84.15 ± 0.53 | 77.29 ± 0.56 | 67.93 ± 0.84 | 60.36 ± 0.77 | 77.19 ± 0.53 | 75.51 ± 0.72 | 76.42 ± 0.65 | 71.02 ± 0.69 |
| DenseNet-121 | Fine-tuned CE (cosine) | 79.09 ± 0.64 | 84.99 ± 0.47 | 72.52 ± 0.59 | 65.52 ± 0.75 | 63.64 ± 0.71 | 67.68 ± 0.56 | 75.30 ± 0.75 | 78.86 ± 0.56 | 73.92 ± 0.56 |
| DenseNet-121 | ProtoNet + MLP head | 77.57 ± 0.69 | 84.03 ± 0.50 | 71.54 ± 0.61 | 61.73 ± 0.75 | 58.43 ± 0.68 | 64.40 ± 0.56 | 72.95 ± 0.74 | 78.87 ± 0.56 | 63.89 ± 0.60 |
| DenseNet-121 | KAN metric + margin (ours) | 78.83 ± 0.67 | 84.75 ± 0.48 | 73.52 ± 0.59 | 62.98 ± 0.73 | 59.72 ± 0.68 | 66.68 ± 0.54 | 76.32 ± 0.79 | 80.52 ± 0.59 | 69.99 ± 0.62 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | 83.91 ± 0.64 | 85.30 ± 0.53 | 82.52 ± 0.52 | 73.94 ± 0.78 | 64.97 ± 0.71 | 83.64 ± 0.44 | 74.93 ± 0.70 | 80.19 ± 0.52 | 66.34 ± 0.63 |
| DINOv2 ViT-S/14 | Fine-tuned CE (cosine) | 82.71 ± 0.61 | 87.23 ± 0.44 | 77.85 ± 0.53 | 63.06 ± 0.73 | 57.63 ± 0.71 | 68.38 ± 0.57 | 76.51 ± 0.73 | 86.68 ± 0.46 | 61.19 ± 0.64 |
| DINOv2 ViT-S/14 | ProtoNet + MLP head | 78.69 ± 0.69 | 85.51 ± 0.52 | 70.33 ± 0.58 | 62.07 ± 0.78 | 53.98 ± 0.70 | 66.00 ± 0.61 | 75.74 ± 0.72 | 80.06 ± 0.53 | 65.54 ± 0.68 |
| DINOv2 ViT-S/14 | KAN metric + margin (ours) | 78.46 ± 0.66 | 85.20 ± 0.48 | 71.16 ± 0.57 | 62.34 ± 0.76 | 55.20 ± 0.71 | 64.15 ± 0.56 | 72.47 ± 0.76 | 80.57 ± 0.57 | 58.60 ± 0.68 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | 85.64 ± 0.60 | 88.01 ± 0.48 | 84.34 ± 0.50 | 73.27 ± 0.77 | 66.37 ± 0.73 | 79.56 ± 0.53 | 73.34 ± 0.71 | 76.71 ± 0.63 | 66.69 ± 0.68 |

## Table 5 – Cross-domain episodes (support from dataset A, query from dataset B), %

Averaged over all ordered dataset pairs and folds where ≥2 novel cultivars are shared; compare with the in-domain numbers of Table 1.

| Backbone | Method | 1-shot | 5-shot |
|---|---|---|---|
| Conv-4 | Fine-tuned CE (cosine) | 50.86 | 53.20 |
| Conv-4 | ProtoNet | 46.73 | 46.09 |
| Conv-4 | ProtoNet + MLP head | 49.78 | 51.40 |
| Conv-4 | ProtoNet + KAN head | 51.09 | 52.36 |
| Conv-4 | + prototype margin | 49.52 | 50.79 |
| Conv-4 | + KAN metric | 49.87 | 50.72 |
| Conv-4 | KAN metric + margin (ours) | 50.42 | 53.18 |
| ResNet-18 | ImageNet features (no training) | 49.86 | 50.06 |
| ResNet-18 | Fine-tuned CE (cosine) | 54.61 | 57.05 |
| ResNet-18 | ProtoNet | 53.67 | 55.31 |
| ResNet-18 | ProtoNet + MLP head | 50.35 | 52.77 |
| ResNet-18 | ProtoNet + KAN head | 50.24 | 51.88 |
| ResNet-18 | + prototype margin | 51.71 | 52.47 |
| ResNet-18 | + KAN metric | 51.37 | 52.76 |
| ResNet-18 | KAN metric + margin (ours) | 51.50 | 53.79 |
| ResNet-18 | ArcFace | 54.72 | 58.07 |
| ResNet-18 | SupCon | 48.52 | 48.55 |
| ResNet-18 | Triplet (batch-hard) | 52.19 | 53.03 |
| ResNet-18 | ProtoNet + triplet | 51.88 | 53.86 |
| ResNet-18 | Ours + KAN head | 50.79 | 52.59 |
| ResNet-50 | ImageNet features (no training) | 55.32 | 59.20 |
| ResNet-50 | Fine-tuned CE (cosine) | 57.44 | 60.47 |
| ResNet-50 | ProtoNet | 54.99 | 56.50 |
| ResNet-50 | ProtoNet + MLP head | 54.03 | 57.09 |
| ResNet-50 | ProtoNet + KAN head | 52.78 | 54.32 |
| ResNet-50 | + prototype margin | 55.01 | 58.05 |
| ResNet-50 | + KAN metric | 54.15 | 55.91 |
| ResNet-50 | KAN metric + margin (ours) | 52.76 | 54.13 |
| DenseNet-121 | ImageNet features (no training) | 57.73 | 62.07 |
| DenseNet-121 | Fine-tuned CE (cosine) | 58.25 | 61.74 |
| DenseNet-121 | ProtoNet | 56.48 | 58.61 |
| DenseNet-121 | ProtoNet + MLP head | 57.09 | 59.31 |
| DenseNet-121 | ProtoNet + KAN head | 57.30 | 60.41 |
| DenseNet-121 | + prototype margin | 54.71 | 57.56 |
| DenseNet-121 | + KAN metric | 55.09 | 57.24 |
| DenseNet-121 | KAN metric + margin (ours) | 54.98 | 57.80 |
| DenseNet-121 | ArcFace | 56.90 | 59.81 |
| DenseNet-121 | SupCon | 51.79 | 53.14 |
| DenseNet-121 | Triplet (batch-hard) | 58.54 | 60.01 |
| DenseNet-121 | ProtoNet + triplet | 55.50 | 58.47 |
| DenseNet-121 | Ours + KAN head | 54.53 | 56.53 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | 60.36 | 65.99 |
| DINOv2 ViT-S/14 | Fine-tuned CE (cosine) | 55.72 | 58.09 |
| DINOv2 ViT-S/14 | ProtoNet | 58.56 | 60.76 |
| DINOv2 ViT-S/14 | ProtoNet + MLP head | 55.65 | 58.16 |
| DINOv2 ViT-S/14 | ProtoNet + KAN head | 57.06 | 59.42 |
| DINOv2 ViT-S/14 | + prototype margin | 50.02 | 51.93 |
| DINOv2 ViT-S/14 | + KAN metric | 54.17 | 56.02 |
| DINOv2 ViT-S/14 | KAN metric + margin (ours) | 53.46 | 55.92 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | 59.44 | 64.50 |
| DINOv2 ViT-B/14 | Fine-tuned CE (cosine) | 53.48 | 53.85 |
| DINOv2 ViT-B/14 | ProtoNet | 59.71 | 60.71 |
| DINOv2 ViT-B/14 | ProtoNet + MLP head | 53.35 | 54.95 |

## Table 6 – Open-set episodes: AUROC (%) for rejecting an unseen cultivar

| Backbone | Method | 1-shot AUROC | 5-shot AUROC | 5-shot closed acc |
|---|---|---|---|---|
| Conv-4 | Fine-tuned CE (cosine) | 70.34 ± 0.63 | 76.30 ± 0.57 | 70.72 |
| Conv-4 | ProtoNet | 70.56 ± 0.69 | 76.82 ± 0.59 | 70.80 |
| Conv-4 | ProtoNet + MLP head | 69.42 ± 0.65 | 75.19 ± 0.58 | 69.41 |
| Conv-4 | ProtoNet + KAN head | 72.82 ± 1.21 | 78.70 ± 1.01 | 72.64 |
| Conv-4 | + prototype margin | 72.79 ± 1.20 | 78.47 ± 1.03 | 71.78 |
| Conv-4 | + KAN metric | 72.20 ± 1.21 | 76.81 ± 1.02 | 70.26 |
| Conv-4 | KAN metric + margin (ours) | 68.83 ± 0.66 | 74.53 ± 0.58 | 68.75 |
| ResNet-18 | ImageNet features (no training) | 68.56 ± 0.66 | 74.79 ± 0.60 | 72.07 |
| ResNet-18 | Fine-tuned CE (cosine) | 73.24 ± 0.62 | 80.46 ± 0.52 | 77.01 |
| ResNet-18 | ProtoNet | 72.10 ± 0.62 | 78.98 ± 0.56 | 73.79 |
| ResNet-18 | ProtoNet + MLP head | 72.87 ± 0.63 | 79.55 ± 0.57 | 75.99 |
| ResNet-18 | ProtoNet + KAN head | 72.43 ± 1.16 | 78.31 ± 1.08 | 73.78 |
| ResNet-18 | + prototype margin | 73.00 ± 1.10 | 79.22 ± 0.98 | 73.69 |
| ResNet-18 | + KAN metric | 72.29 ± 1.23 | 77.31 ± 1.18 | 74.06 |
| ResNet-18 | KAN metric + margin (ours) | 72.42 ± 0.65 | 79.11 ± 0.59 | 75.28 |
| ResNet-18 | ArcFace | 72.28 ± 1.19 | 77.92 ± 1.01 | 73.93 |
| ResNet-18 | SupCon | 70.26 ± 1.22 | 74.79 ± 1.09 | 69.62 |
| ResNet-18 | Triplet (batch-hard) | 70.41 ± 1.19 | 76.62 ± 1.02 | 71.02 |
| ResNet-18 | ProtoNet + triplet | 70.83 ± 1.17 | 76.82 ± 1.01 | 71.99 |
| ResNet-18 | Ours + KAN head | 71.58 ± 1.15 | 77.02 ± 1.12 | 72.62 |
| ResNet-50 | ImageNet features (no training) | 68.00 ± 0.68 | 74.95 ± 0.60 | 72.56 |
| ResNet-50 | Fine-tuned CE (cosine) | 73.04 ± 0.64 | 80.00 ± 0.57 | 76.93 |
| ResNet-50 | ProtoNet | 73.37 ± 0.62 | 80.34 ± 0.53 | 76.47 |
| ResNet-50 | ProtoNet + MLP head | 73.88 ± 0.64 | 80.72 ± 0.55 | 76.98 |
| ResNet-50 | ProtoNet + KAN head | 70.94 ± 1.12 | 76.24 ± 1.03 | 73.20 |
| ResNet-50 | + prototype margin | 71.00 ± 1.12 | 76.09 ± 1.06 | 72.47 |
| ResNet-50 | + KAN metric | 72.05 ± 1.18 | 77.14 ± 1.07 | 74.24 |
| ResNet-50 | KAN metric + margin (ours) | 74.12 ± 0.63 | 80.77 ± 0.54 | 77.24 |
| DenseNet-121 | ImageNet features (no training) | 67.44 ± 0.66 | 74.66 ± 0.57 | 72.09 |
| DenseNet-121 | Fine-tuned CE (cosine) | 74.32 ± 0.62 | 81.03 ± 0.55 | 78.72 |
| DenseNet-121 | ProtoNet | 73.23 ± 0.65 | 80.12 ± 0.58 | 77.21 |
| DenseNet-121 | ProtoNet + MLP head | 73.47 ± 0.63 | 80.09 ± 0.57 | 77.51 |
| DenseNet-121 | ProtoNet + KAN head | 72.66 ± 1.12 | 78.35 ± 1.06 | 75.87 |
| DenseNet-121 | + prototype margin | 72.25 ± 1.10 | 78.22 ± 1.02 | 75.02 |
| DenseNet-121 | + KAN metric | 71.84 ± 1.20 | 77.57 ± 1.11 | 76.19 |
| DenseNet-121 | KAN metric + margin (ours) | 73.47 ± 0.62 | 80.45 ± 0.55 | 77.75 |
| DenseNet-121 | ArcFace | 70.17 ± 1.16 | 75.57 ± 1.01 | 71.65 |
| DenseNet-121 | SupCon | 67.39 ± 1.19 | 70.92 ± 1.10 | 65.58 |
| DenseNet-121 | Triplet (batch-hard) | 71.30 ± 1.15 | 77.38 ± 1.01 | 72.73 |
| DenseNet-121 | ProtoNet + triplet | 72.45 ± 1.07 | 78.65 ± 0.95 | 74.08 |
| DenseNet-121 | Ours + KAN head | 71.06 ± 1.16 | 76.14 ± 1.12 | 74.56 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | 69.00 ± 0.56 | 77.11 ± 0.46 | 76.12 |
| DINOv2 ViT-S/14 | Fine-tuned CE (cosine) | 75.58 ± 0.60 | 82.50 ± 0.53 | 78.77 |
| DINOv2 ViT-S/14 | ProtoNet | 74.91 ± 0.61 | 81.06 ± 0.55 | 77.28 |
| DINOv2 ViT-S/14 | ProtoNet + MLP head | 74.74 ± 0.61 | 81.33 ± 0.54 | 77.38 |
| DINOv2 ViT-S/14 | ProtoNet + KAN head | 74.07 ± 1.21 | 79.16 ± 1.10 | 75.03 |
| DINOv2 ViT-S/14 | + prototype margin | 73.57 ± 1.05 | 79.13 ± 0.99 | 73.72 |
| DINOv2 ViT-S/14 | + KAN metric | 72.33 ± 1.17 | 76.99 ± 1.10 | 74.26 |
| DINOv2 ViT-S/14 | KAN metric + margin (ours) | 74.62 ± 0.62 | 80.97 ± 0.54 | 77.28 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | 70.45 ± 0.53 | 78.33 ± 0.45 | 75.96 |
| DINOv2 ViT-B/14 | Fine-tuned CE (cosine) | 76.82 ± 0.58 | 83.16 ± 0.50 | 79.39 |
| DINOv2 ViT-B/14 | ProtoNet | 76.74 ± 0.57 | 83.16 ± 0.49 | 78.50 |
| DINOv2 ViT-B/14 | ProtoNet + MLP head | 76.02 ± 0.61 | 81.90 ± 0.55 | 77.93 |

## Table 7 – Sensitivity and ablations (ResNet-18 unless stated), 1/5-shot %

| Setting | 1-shot | 5-shot |
|---|---|---|
| KAN metric, margin m=0.0 | 63.73 ± 0.50 | 73.98 ± 0.43 |
| KAN metric, margin m=0.05 | 63.01 ± 0.49 | 73.35 ± 0.43 |
| KAN metric, margin m=0.1 | 62.07 ± 0.49 | 72.87 ± 0.42 |
| KAN metric, margin m=0.2 | 61.51 ± 0.49 | 71.79 ± 0.43 |
| KAN metric, margin m=0.4 | 61.76 ± 0.48 | 72.56 ± 0.42 |
| Triplet, random mining | 62.78 ± 0.50 | 73.16 ± 0.42 |
| Triplet, semi_hard mining | 61.25 ± 0.49 | 71.13 ± 0.43 |
| Triplet, batch_hard mining | 60.33 ± 0.50 | 70.78 ± 0.43 |
| Triplet, distance_weighted mining | 61.20 ± 0.50 | 71.69 ± 0.43 |
| ResNet-18, ProtoNet + MLP head, fruit crop | 63.78 ± 0.50 | 74.34 ± 0.42 |
| ResNet-18, ProtoNet + MLP head, full frame (background kept) | 62.23 ± 0.49 | 72.53 ± 0.42 |
| ResNet-18, KAN metric + margin (ours), fruit crop | 62.07 ± 0.49 | 72.87 ± 0.42 |
| ResNet-18, KAN metric + margin (ours), full frame (background kept) | 61.62 ± 0.50 | 72.36 ± 0.42 |
| DenseNet-121, ProtoNet + MLP head, fruit crop | 64.81 ± 0.48 | 75.17 ± 0.40 |
| DenseNet-121, ProtoNet + MLP head, full frame (background kept) | 65.04 ± 0.47 | 75.22 ± 0.40 |
| DenseNet-121, KAN metric + margin (ours), fruit crop | 64.31 ± 0.48 | 75.63 ± 0.40 |
| DenseNet-121, KAN metric + margin (ours), full frame (background kept) | 66.17 ± 0.48 | 75.81 ± 0.40 |
| ResNet-18, ProtoNet + MLP head: 5-shot over seeds [0, 1, 2] | – | 74.12 ± 0.57 (sd) |
| ResNet-18, KAN metric + margin (ours): 5-shot over seeds [0, 1, 2] | – | 73.57 ± 0.62 (sd) |
| DenseNet-121, ProtoNet + MLP head: 5-shot over seeds [0, 1, 2] | – | 75.22 ± 0.27 (sd) |
| DenseNet-121, KAN metric + margin (ours): 5-shot over seeds [0, 1, 2] | – | 74.77 ± 0.79 (sd) |

## Table 8 – Closed-set limited data: normal fine-tuning vs few-shot prototypes (macro-F1 %)

All cultivars known; K training images per cultivar; test = held-out capture groups. Mean over 3 draws of the K images.  `proto_meta[...]` uses an embedding meta-trained on the *other two* datasets (LODO checkpoint) and only the K shots of the target dataset.


**Mangifera2012 – DenseNet-121**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 52.6 | 82.0 | 90.9 | 94.3 | 97.0 |
| finetune_imagenet | mlp | 56.2 | 81.8 | 91.6 | 95.2 | 97.4 |
| finetune_meta[ours] | mlp | 58.3 | 84.7 | 92.6 | 94.8 | 97.6 |
| finetune_meta[proto_mlp] | mlp | 58.2 | 83.6 | 92.0 | 95.8 | 97.6 |
| proto_imagenet | none | 48.8 | 76.0 | 83.3 | 85.0 | 85.2 |
| proto_meta[ours] | mlp | 54.3 | 71.5 | 75.1 | 77.7 | 79.7 |
| proto_meta[proto_mlp] | mlp | 46.3 | 65.1 | 69.0 | 70.5 | 72.2 |

**Mangifera2012 – ResNet-18**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 49.0 | 79.7 | 89.7 | 93.6 | 97.8 |
| finetune_imagenet | mlp | 48.1 | 79.7 | 89.6 | 94.0 | 97.0 |
| finetune_meta[ours] | mlp | 51.4 | 81.7 | 91.7 | 94.1 | 97.5 |
| finetune_meta[proto_mlp] | mlp | 54.4 | 84.3 | 92.0 | 94.9 | 97.8 |
| proto_imagenet | none | 51.2 | 74.8 | 81.0 | 82.9 | 84.8 |
| proto_meta[ours] | mlp | 41.2 | 63.5 | 68.5 | 73.0 | 74.0 |
| proto_meta[proto_mlp] | mlp | 45.0 | 58.6 | 66.0 | 67.8 | 69.4 |

**MangoClassify12 – DenseNet-121**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 30.7 | 58.1 | 70.5 | 81.6 | 96.6 |
| finetune_imagenet | mlp | 34.9 | 59.1 | 70.7 | 80.9 | 94.6 |
| finetune_meta[ours] | mlp | 36.1 | 60.7 | 73.3 | 80.0 | 96.4 |
| finetune_meta[proto_mlp] | mlp | 39.1 | 59.2 | 70.6 | 79.1 | 96.4 |
| proto_imagenet | none | 35.1 | 54.3 | 59.3 | 63.5 | 67.3 |
| proto_meta[ours] | mlp | 33.7 | 44.2 | 50.7 | 55.8 | 56.2 |
| proto_meta[proto_mlp] | mlp | 30.0 | 44.3 | 49.2 | 56.3 | 55.7 |

**MangoClassify12 – ResNet-18**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 30.6 | 57.2 | 66.8 | 74.6 | 94.7 |
| finetune_imagenet | mlp | 31.0 | 56.4 | 68.5 | 74.9 | 95.0 |
| finetune_meta[ours] | mlp | 37.7 | 60.0 | 70.2 | 78.7 | 91.8 |
| finetune_meta[proto_mlp] | mlp | 35.5 | 60.9 | 70.6 | 78.3 | 94.8 |
| proto_imagenet | none | 31.7 | 54.8 | 59.8 | 64.2 | 66.4 |
| proto_meta[ours] | mlp | 24.6 | 43.9 | 48.5 | 54.0 | 57.1 |
| proto_meta[proto_mlp] | mlp | 25.1 | 43.8 | 50.9 | 55.4 | 58.2 |

**MangoImageBD – DenseNet-121**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 38.9 | 66.6 | 78.0 | 87.1 | 98.8 |
| finetune_imagenet | mlp | 39.9 | 71.0 | 78.4 | 87.8 | 98.0 |
| finetune_meta[ours] | mlp | 47.4 | 73.7 | 82.8 | 90.5 | 98.7 |
| finetune_meta[proto_mlp] | mlp | 42.7 | 73.0 | 81.6 | 89.1 | 98.1 |
| proto_imagenet | none | 42.2 | 61.9 | 65.4 | 68.0 | 70.0 |
| proto_meta[ours] | mlp | 37.3 | 50.9 | 56.3 | 58.6 | 59.6 |
| proto_meta[proto_mlp] | mlp | 34.2 | 50.3 | 53.4 | 57.0 | 57.2 |

**MangoImageBD – ResNet-18**

| Method | head | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|---|
| finetune_imagenet | kan | 37.0 | 63.2 | 77.5 | 86.3 | 98.7 |
| finetune_imagenet | mlp | 34.7 | 64.9 | 76.9 | 85.1 | 98.1 |
| finetune_meta[ours] | mlp | 40.1 | 71.8 | 79.4 | 88.6 | 97.9 |
| finetune_meta[proto_mlp] | mlp | 40.8 | 71.2 | 79.2 | 88.9 | 98.3 |
| proto_imagenet | none | 41.9 | 60.2 | 63.0 | 67.2 | 69.6 |
| proto_meta[ours] | mlp | 30.9 | 46.9 | 49.0 | 52.3 | 52.5 |
| proto_meta[proto_mlp] | mlp | 32.0 | 48.5 | 50.6 | 52.3 | 52.5 |

## Table 9 – Leakage inflation: random image split vs capture-group split (K=all, fine-tuned, acc %)

| Dataset | Backbone | head | group split | random split | inflation |
|---|---|---|---|---|---|
| Mangifera2012 | DenseNet-121 | kan | 97.12 | 99.67 | +2.55 |
| Mangifera2012 | DenseNet-121 | mlp | 97.44 | 99.50 | +2.07 |
| Mangifera2012 | ResNet-18 | kan | 97.76 | 98.68 | +0.92 |
| Mangifera2012 | ResNet-18 | mlp | 96.96 | 98.84 | +1.89 |
| MangoClassify12 | DenseNet-121 | kan | 97.50 | 98.38 | +0.88 |
| MangoClassify12 | DenseNet-121 | mlp | 96.58 | 98.12 | +1.54 |
| MangoClassify12 | ResNet-18 | kan | 96.33 | 97.27 | +0.94 |
| MangoClassify12 | ResNet-18 | mlp | 95.83 | 96.84 | +1.01 |
| MangoImageBD | DenseNet-121 | kan | 99.53 | 99.01 | -0.53 |
| MangoImageBD | DenseNet-121 | mlp | 99.19 | 99.01 | -0.18 |
| MangoImageBD | ResNet-18 | kan | 99.30 | 98.77 | -0.53 |
| MangoImageBD | ResNet-18 | mlp | 99.07 | 99.18 | +0.11 |

## Table 10 – Per-episode full fine-tuning ("normal fine-tuning") on novel cultivars, %

ImageNet backbone + fresh classifier fine-tuned on each support set for 100 steps (100 episodes per fold).  Compare with prototype inference in Table 1.

| Backbone | fold | 1-shot | 5-shot | s / episode |
|---|---|---|---|---|
| ResNet-18 | 0 | 62.37 ± 1.79 | 75.97 ± 1.88 | 6.0 |
| ResNet-18 | 1 | 60.89 ± 2.10 | 76.91 ± 1.59 | 6.0 |
| ResNet-18 | 2 | 64.24 ± 2.15 | 83.51 ± 1.54 | 6.0 |
| DenseNet-121 | 0 | 64.21 ± 1.91 | 76.51 ± 1.81 | 16.7 |
| DenseNet-121 | 1 | 62.76 ± 2.01 | 78.25 ± 1.49 | 16.6 |
| DenseNet-121 | 2 | 66.24 ± 2.07 | 85.57 ± 1.50 | 16.6 |

## Table 11 – Cost (T4 GPU fp16; CPU 4 threads)

| backbone    | head   | metric   |   params_backbone_M |   params_head_M |   params_metric_M |   gmacs |   gpu_ms_b1 |   gpu_ms_b32 |   cpu_ms_b1 |   gpu_peak_mem_mb_b32 |   metric_ms_75x5 |
|:------------|:-------|:---------|--------------------:|----------------:|------------------:|--------:|------------:|-------------:|------------:|----------------------:|-----------------:|
| conv4       | none   | euclid   |               0.113 |           0     |             0     |   0.097 |       0.704 |        3.551 |       4.705 |                68.585 |          nan     |
| conv4       | mlp    | euclid   |               0.113 |           0.446 |             0     |   0.098 |       0.897 |        3.239 |       6.157 |                79.414 |          nan     |
| conv4       | kan    | euclid   |               0.113 |           4.427 |             0     |   0.101 |       2.258 |        3.258 |       8.722 |                94.685 |          nan     |
| conv4       | mlp    | kan      |               0.113 |           0.446 |             0.001 |   0.098 |       1.061 |        2.314 |       4.935 |                79.424 |            0.886 |
| resnet18    | none   | euclid   |              11.177 |           0     |             0     |   1.814 |       2.628 |       12.856 |      30.701 |               211.739 |          nan     |
| resnet18    | mlp    | euclid   |              11.177 |           0.166 |             0     |   1.814 |       2.84  |       12.91  |      34.508 |               212.184 |          nan     |
| resnet18    | kan    | euclid   |              11.177 |           1.64  |             0     |   1.815 |       4.603 |       13.338 |      33.128 |               217.842 |          nan     |
| resnet18    | mlp    | kan      |              11.177 |           0.166 |             0.001 |   1.814 |       2.875 |       13.007 |      32.962 |               212.194 |            0.788 |
| resnet50    | none   | euclid   |              23.508 |           0     |             0     |   4.087 |       6.898 |       37.645 |      85.844 |               310.505 |          nan     |
| resnet50    | mlp    | euclid   |              23.508 |           0.562 |             0     |   4.088 |       6.797 |       37.736 |      78.094 |               312.649 |          nan     |
| resnet50    | kan    | euclid   |              23.508 |           5.575 |             0     |   4.092 |       9.592 |       38.711 |      81.903 |               332.128 |          nan     |
| resnet50    | mlp    | kan      |              23.508 |           0.562 |             0.001 |   4.088 |       6.645 |       37.876 |      77.191 |               312.66  |            0.814 |
| densenet121 | none   | euclid   |               6.954 |           0     |             0     |   2.833 |      16.047 |       44.325 |      91.295 |               227.104 |          nan     |
| densenet121 | mlp    | euclid   |               6.954 |           0.298 |             0     |   2.833 |      16.968 |       44.253 |      92.426 |               228.241 |          nan     |
| densenet121 | kan    | euclid   |               6.954 |           2.952 |             0     |   2.836 |      18.397 |       44.853 |      93.881 |               239.423 |          nan     |
| densenet121 | mlp    | kan      |               6.954 |           0.298 |             0.001 |   2.833 |      16.832 |       44.416 |      89.752 |               228.251 |            0.83  |
| dinov2_s    | none   | euclid   |              21.629 |           0     |             0     |   5.515 |       6.209 |       35.37  |      86.467 |               270.101 |          nan     |
| dinov2_s    | mlp    | euclid   |              21.629 |           0.133 |             0     |   5.515 |       6.232 |       35.39  |      80.237 |               270.607 |          nan     |
| dinov2_s    | kan    | euclid   |              21.629 |           1.312 |             0     |   5.516 |       7.753 |       35.793 |      84.915 |               275.51  |          nan     |
| dinov2_s    | mlp    | kan      |              21.629 |           0.133 |             0.001 |   5.515 |       6.469 |       35.401 |      87.689 |               270.618 |            0.86  |
| dinov2_b    | none   | euclid   |              85.725 |           0     |             0     |  21.944 |      10.943 |       95.06  |     277.162 |               739.605 |          nan     |
| dinov2_b    | mlp    | euclid   |              85.725 |           0.232 |             0     |  21.944 |      10.026 |       95.193 |     287.774 |               737.802 |          nan     |
| dinov2_b    | kan    | euclid   |              85.725 |           2.296 |             0     |  21.946 |      11.54  |       95.953 |     292.096 |               743.582 |          nan     |
| dinov2_b    | mlp    | kan      |              85.725 |           0.232 |             0.001 |  21.944 |      11.394 |       95.442 |     285.042 |               737.813 |            0.789 |

## Table 12 – Background removal, lightweight CNNs and feature fusion, %

Nearest prototype on the backbone feature f; LR = logistic regression on z; cross = 5-shot cross-domain episodes (as Table 5); AUROC = 5-shot open set.

| Backbone | Model | Images | 1-shot | 5-shot | 10-shot | 5-shot LR | cross | AUROC |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | ImageNet features (no training) | crop | 63.3 | 75.3 | 79.1 | 77.5 | 50.1 | 75.5 |
| ResNet-18 | ImageNet features (no training) | background removed | 60.2 | 73.3 | 77.5 | 75.6 | 50.9 | 72.8 |
| ResNet-50 | ImageNet features (no training) | crop | 63.5 | 75.0 | 78.3 | 78.2 | 59.2 | 75.3 |
| ResNet-50 | ImageNet features (no training) | background removed | 60.1 | 72.3 | 76.0 | 75.7 | 57.7 | 73.5 |
| DenseNet-121 | ImageNet features (no training) | crop | 62.1 | 74.7 | 78.5 | 77.9 | 62.1 | 75.0 |
| DenseNet-121 | ImageNet features (no training) | background removed | 58.8 | 72.3 | 76.3 | 75.3 | 63.4 | 71.9 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | crop | 63.9 | 79.4 | 83.9 | 81.7 | 66.0 | 77.7 |
| DINOv2 ViT-S/14 | ImageNet features (no training) | background removed | 59.8 | 76.1 | 81.2 | 79.1 | 65.6 | 75.0 |
| DINOv2 ViT-B/14 | Fine-tuned CE (cosine) | crop | 74.8 | 84.0 | 86.7 | 79.2 | 64.4 | 85.7 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | crop | 64.4 | 79.1 | 83.2 | 82.2 | 64.5 | 79.0 |
| DINOv2 ViT-B/14 | ImageNet features (no training) | background removed | 62.1 | 76.8 | 81.3 | 80.1 | 64.5 | 77.4 |
| EfficientNet-B0 | ImageNet features (no training) | crop | 62.8 | 75.9 | 79.6 | 78.8 | 57.0 | 75.0 |
| EfficientNet-B0 | ImageNet features (no training) | background removed | 59.5 | 74.2 | 78.6 | 77.4 | 62.2 | 73.5 |
| MobileNetV2 | ImageNet features (no training) | crop | 64.0 | 76.1 | 79.8 | 78.8 | 52.6 | 77.0 |
| MobileNetV2 | ImageNet features (no training) | background removed | 60.4 | 73.7 | 78.2 | 76.6 | 52.4 | 74.3 |
| Fusion (5 backbones) | fusion of 5 backbones, ce | crop | 76.1 | 85.0 | 87.6 | 86.3 | 63.5 | 85.9 |
| Fusion (5 backbones) | fusion of 5 backbones, imagenet | crop | 69.7 | 81.7 | 85.1 | 84.3 | 63.5 | 81.1 |
