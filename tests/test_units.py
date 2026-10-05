"""Unit tests for the parts where a silent bug would invalidate the paper.

run:  .venv/bin/python -m pytest -q tests/test_units.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import (build_groups, cultivar_folds, eligible_classes, protocol_split,  # noqa: E402
                  sample_episode, gpu_augment)
from evaluate import auroc, proto_predict, macro_f1, ci95  # noqa: E402
from losses import proto_loss, triplet_loss, supcon_loss, pdist2, Objective  # noqa: E402
from models import KANLinear, KANMetric, KANHead, MLPHead  # noqa: E402
from prep_data import canon  # noqa: E402


# ---------------------------------------------------------------- fixtures
def fake_meta(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    cultivars = {"MangoImageBD": ["Amrapali", "Fazli", "Langra", "Shada", "Rupali", "Gourmoti"],
                 "MangoClassify12": ["Amrapali", "Fazli", "Sundari", "RaniBhog", "GopalBhog"],
                 "Mangifera2012": ["Amrapali", "Langra", "Mollika", "Nilambori", "Bari7"]}
    for ds, cs in cultivars.items():
        for c in cs:
            for i in range(n_per):
                rows.append(dict(dataset=ds, cultivar=c, phash=f"{rng.integers(1 << 62):016x}",
                                 exif_time="", camera=""))
    m = pd.DataFrame(rows)
    m["idx"] = np.arange(len(m))
    m["group"] = np.arange(len(m)) // 3      # groups of 3 photos of one fruit
    m["drop"] = 0
    return m


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("raw,expected", [
    ("Amrapali-252", "Amrapali"), ("Bari-7-176", "Bari7"), ("Bari 4", "Bari4"), ("Bari-4", "Bari4"),
    ("Fazlee-156", "Fazli"), ("Fazli Classic", "Fazli"), ("Himsagor", "Himsagar"),
    ("Kanchon Langra-210", "KanchonLangra"), ("Banana Mango", "Banana"), ("Bari-11", "Bari11"),
])
def test_canonical_names(raw, expected):
    assert canon(raw) == expected


def test_unknown_name_fails_loudly():
    with pytest.raises(KeyError):
        canon("NotAMango")


# ----------------------------------------------------------------- splits
def test_unified_folds_partition_cultivars():
    m = fake_meta()
    folds = cultivar_folds(m, 3)
    flat = sum(folds, [])
    assert sorted(flat) == sorted(m.cultivar.unique())        # every cultivar exactly once
    for f in range(3):
        tr, te, info = protocol_split(m, "unified", f)
        assert not set(m.loc[tr, "cultivar"]) & set(m.loc[te, "cultivar"])   # cultivar-disjoint
        assert len(set(tr) & set(te)) == 0


def test_lodo_split():
    m = fake_meta()
    tr, te, info = protocol_split(m, "lodo", 2)
    assert set(m.loc[te, "dataset"]) == {"Mangifera2012"}
    assert "Mangifera2012" not in set(m.loc[tr, "dataset"])
    assert set(info["seen"]) == {"Amrapali", "Langra"}
    assert set(info["novel"]) == {"Mollika", "Nilambori", "Bari7"}


def test_dropped_images_excluded():
    m = fake_meta()
    m.loc[:9, "drop"] = 1
    tr, te, _ = protocol_split(m, "unified", 0)
    assert not set(range(10)) & (set(tr) | set(te))


# --------------------------------------------------------------- episodes
def test_episode_support_query_group_disjoint():
    m = fake_meta()
    labels = m.cultivar.astype("category").cat.codes.to_numpy()
    groups = m.group.to_numpy()
    rng = np.random.default_rng(0)
    classes = eligible_classes(labels, groups, 5, 15)
    for _ in range(200):
        sp, ys, qp, yq, cls = sample_episode(rng, labels, groups, classes, 5, 5, 15)
        assert not set(groups[sp]) & set(groups[qp])           # no fruit in both
        assert len(set(sp) & set(qp)) == 0
        assert (labels[sp] == cls[ys]).all() and (labels[qp] == cls[yq]).all()
        assert np.bincount(ys).tolist() == [5] * 5


def test_episodes_reproducible():
    m = fake_meta()
    labels = m.cultivar.astype("category").cat.codes.to_numpy()
    groups = m.group.to_numpy()
    classes = eligible_classes(labels, groups, 1, 15)
    a = sample_episode(np.random.default_rng(7), labels, groups, classes, 5, 1, 15)
    b = sample_episode(np.random.default_rng(7), labels, groups, classes, 5, 1, 15)
    assert all((x == y).all() for x, y in zip(a, b))


def test_grouping_links_near_duplicates_and_drops_cross_label():
    m = fake_meta(n_per=4)
    m["phash"] = [f"{i * 0x1111111111:016x}" for i in range(len(m))]
    m.loc[1, "phash"] = m.loc[0, "phash"]                     # same image twice, same label
    j = m.index[m.cultivar != m.cultivar.iat[0]][0]
    m.loc[j, "phash"] = m.loc[2, "phash"]                     # same image, different label
    m.loc[0, ["exif_time", "camera"]] = ["2025:06:27 18:00:00", "cam"]
    m.loc[3, ["exif_time", "camera"]] = ["2025:06:27 18:00:10", "cam"]  # 10 s burst
    g, drop, stats = build_groups(m, None, phash_thr=0)
    assert g[0] == g[1] == g[3]
    assert drop[2] == 1 and drop[j] == 1
    assert stats["cross_label_dups"] >= 1


# ------------------------------------------------------------------- KAN
def test_kan_spline_partition_of_unity():
    layer = KANLinear(3, 2, grid_size=5, grid_range=(-2, 2))
    x = torch.linspace(-1.99, 1.99, 50).unsqueeze(1).repeat(1, 3)
    b = layer.b_splines(x)
    assert torch.allclose(b.sum(-1), torch.ones_like(b.sum(-1)), atol=1e-5)


def test_kan_metric_starts_as_euclidean():
    torch.manual_seed(0)
    met = KANMetric(16)
    q, c = F.normalize(torch.randn(7, 16), dim=1), F.normalize(torch.randn(4, 16), dim=1)
    d = met(q, c)
    assert d.shape == (7, 4)
    euc = pdist2(q, c)
    assert torch.allclose(d.argmin(1), euc.argmin(1)) or (d - euc).abs().max() < 0.05


def test_kan_metric_and_heads_get_gradients():
    met, kh, mh = KANMetric(8), KANHead(32, 16, 8), MLPHead(32, 16, 8)
    x = torch.randn(10, 32)
    z = kh(x) + mh(x)
    met(F.normalize(z, dim=1), F.normalize(z[:3], dim=1)).sum().backward()
    for mod in (met, kh, mh):
        assert all(p.grad is not None for p in mod.parameters() if p.requires_grad)


# ----------------------------------------------------------------- losses
def test_proto_margin_increases_loss():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(40, 16), dim=1)
    y = torch.arange(4).repeat_interleave(10)
    l0, a0 = proto_loss(z, y, 5, 10.0)
    l1, a1 = proto_loss(z, y, 5, 10.0, margin=0.2)
    assert l1 > l0 and a0 == a1                               # margin changes loss, not accuracy


def test_perfect_embedding_has_low_losses():
    y = torch.arange(4).repeat_interleave(6)
    z = F.one_hot(y, 8).float()
    assert triplet_loss(z, y, mining="batch_hard") == 0
    l, acc = proto_loss(z, y, 3, 10.0)
    assert acc == 1 and l < 1e-3
    assert supcon_loss(z, y) < supcon_loss(torch.randn(24, 8), y)


@pytest.mark.parametrize("mining", ["random", "semi_hard", "batch_hard", "distance_weighted"])
def test_triplet_mining_finite(mining):
    torch.manual_seed(1)
    z = F.normalize(torch.randn(40, 16, requires_grad=True), dim=1)
    y = torch.arange(8).repeat_interleave(5)
    loss = triplet_loss(z, y, mining=mining)
    assert torch.isfinite(loss)


def test_objective_uses_model_metric():
    obj = Objective("proto", 8, 4, n_support=3)
    assert not any(n.startswith("metric") for n, _ in obj.named_parameters())  # metric not duplicated
    z = torch.randn(24, 8)
    y = torch.arange(4).repeat_interleave(6)
    loss, _ = obj(z, y, KANMetric(8))
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ eval
def test_auroc_matches_definition():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, 300)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(auroc(pos, neg) - brute) < 1e-9
    assert auroc([1, 1], [1, 1]) == 0.5


def test_proto_predict_and_f1():
    es = torch.eye(3).repeat_interleave(2, 0)
    ys = torch.arange(3).repeat_interleave(2)
    pred, prob = proto_predict(es, ys, torch.eye(3), 3)
    assert pred.tolist() == [0, 1, 2]
    assert macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2]), 3) == 1.0
    m, h = ci95([0.5] * 10)
    assert m == 0.5 and h == 0.0


def test_gpu_augment_shapes_and_range():
    x = torch.rand(4, 3, 288, 288)
    for train in (True, False):
        y = gpu_augment(x, 224, train=train)
        assert y.shape == (4, 3, 224, 224) and y.min() >= 0 and y.max() <= 1
