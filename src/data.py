"""Image cache, leakage-safe grouping, protocol splits, samplers, GPU augmentation."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DATASETS = ["MangoImageBD", "MangoClassify12", "Mangifera2012"]


class Cache:
    """uint8 image array (N,S,S,3) kept in RAM + metadata dataframe."""

    def __init__(self, root, variant="crop", groups_path=None, in_memory=True):
        self.root = root
        self.meta = pd.read_csv(os.path.join(root, "meta.csv"))
        gpath = groups_path or os.path.join(root, "groups.csv")
        if os.path.exists(gpath):
            g = pd.read_csv(gpath)
            self.meta = self.meta.merge(g[["idx", "group", "drop"]], on="idx", how="left")
        self.images = np.load(os.path.join(root, f"{variant}.npy"), mmap_mode="r")
        if in_memory:  # training reads random batches; keep the array in RAM
            self.images = np.ascontiguousarray(self.images)
        assert len(self.images) == len(self.meta)

    def batch(self, idx, device):
        x = torch.from_numpy(self.images[np.asarray(idx)]).to(device, non_blocking=True)
        return x.permute(0, 3, 1, 2).float().div_(255)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _parse_time(t):
    try:
        return datetime.strptime(str(t)[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        return None


def build_groups(meta, feats=None, seq_gap_s=10, dup_cos=0.97, xlabel_dup_cos=0.97, phash_thr=2):
    """Capture-group ids used to keep related photos on one side of every split.

    Two images share a group (within one dataset and cultivar) if
      * they belong to the same capture sequence: same camera, consecutive
        EXIF timestamps <= seq_gap_s apart (multi-view shots of a fruit), or
      * they are near-duplicates: complete-linkage clusters of DINOv2
        embeddings with every pairwise cosine >= dup_cos (complete linkage
        avoids the chaining that single linkage produces among same-cultivar
        fruits, which are all fairly similar), or
      * their perceptual hashes differ by <= phash_thr bits AND (if features
        are given) cosine >= dup_cos (re-encoded copies).
    An image whose near-copy (cosine >= xlabel_dup_cos, or identical pHash)
    carries a *different* cultivar label is marked drop=1 (label conflict).

    Note: pHash alone is unreliable here - centred fruit on plain backgrounds
    gives many low-Hamming pairs across cultivars (see docs/grouping.md).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(meta)
    uf = UnionFind(n)
    drop = np.zeros(n, dtype=int)
    stats = {"seq_links": 0, "dup_links": 0, "phash_links": 0, "cross_label_dups": 0,
             "cross_dataset_dups": 0}
    f = None if feats is None else F.normalize(torch.as_tensor(feats).float(), dim=1)
    ds = meta["dataset"].to_numpy()
    cv = meta["cultivar"].to_numpy()

    # 1) capture sequences
    t = meta["exif_time"].map(_parse_time)
    key = meta["dataset"] + "|" + meta["cultivar"] + "|" + meta["camera"].fillna("")
    df = pd.DataFrame({"t": t, "key": key, "i": np.arange(n)}).dropna(subset=["t"])
    for _, g in df.groupby("key"):
        g = g.sort_values("t")
        ts, ids = g["t"].to_numpy(), g["i"].to_numpy()
        for k in range(1, len(ts)):
            if ts[k] - ts[k - 1] <= seq_gap_s:
                uf.union(int(ids[k - 1]), int(ids[k]))
                stats["seq_links"] += 1

    # 2) near-duplicate clusters (complete linkage) per dataset x cultivar
    if f is not None:
        key2 = meta["dataset"] + "|" + meta["cultivar"]
        for _, ids in meta.groupby(key2).indices.items():
            if len(ids) < 2:
                continue
            D = (1 - f[ids] @ f[ids].t()).clamp_min(0).numpy().astype(np.float64)
            np.fill_diagonal(D, 0)
            lab = fcluster(linkage(squareform(D, checks=False), "complete"), 1 - dup_cos, "distance")
            for c in np.unique(lab):
                mem = ids[lab == c]
                for j in mem[1:]:
                    uf.union(int(mem[0]), int(j))
                    stats["dup_links"] += 1
        # label conflicts: near-identical images under different cultivars
        for s0 in range(0, n, 2048):
            sim = f[s0:s0 + 2048] @ f.t()
            ii, jj = (sim >= xlabel_dup_cos).nonzero(as_tuple=True)
            for i, j in zip((ii + s0).tolist(), jj.tolist()):
                if i < j and cv[i] != cv[j]:
                    drop[i] = drop[j] = 1
                    stats["cross_label_dups"] += 1
                if i < j and ds[i] != ds[j]:
                    stats["cross_dataset_dups"] += 1

    # 3) re-encoded copies via pHash
    h = np.array([int(x, 16) for x in meta["phash"]], dtype=np.uint64)
    B = torch.from_numpy(np.unpackbits(h.view(np.uint8).reshape(n, 8), axis=1).astype(np.float32))
    for s0 in range(0, n, 2048):
        ham = B[s0:s0 + 2048] @ (1 - B).t() + (1 - B[s0:s0 + 2048]) @ B.t()
        ii, jj = (ham <= phash_thr).nonzero(as_tuple=True)
        for i, j in zip((ii + s0).tolist(), jj.tolist()):
            if i >= j:
                continue
            if f is not None and float(f[i] @ f[j]) < dup_cos:
                continue
            if cv[i] != cv[j]:
                if f is None and ham[i - s0, j] == 0:
                    drop[i] = drop[j] = 1
                    stats["cross_label_dups"] += 1
                continue
            if ds[i] == ds[j]:
                uf.union(i, j)
                stats["phash_links"] += 1
    roots = np.array([uf.find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group, drop, stats


def cultivar_folds(meta, n_folds=3, seed=0):
    """Assign every cultivar to exactly one novel fold.

    Cultivars are sorted by image count and dealt round-robin (snake order)
    so every fold gets a similar mix of large and small cultivars."""
    counts = meta.groupby("cultivar").size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    names = list(counts.index)
    folds = [[] for _ in range(n_folds)]
    for r in range(0, len(names), n_folds):
        chunk = names[r:r + n_folds]
        order = rng.permutation(n_folds) if r // n_folds % 2 == 0 else rng.permutation(n_folds)[::-1]
        for name, f in zip(chunk, order):
            folds[f].append(name)
    return [sorted(f) for f in folds]


def protocol_split(meta, protocol, fold, seed=0):
    """Return (train_idx, test_idx, info) for a protocol/fold.

    unified : cultivar-disjoint 3-fold CV over the merged 24-cultivar pool;
              train = all images of base cultivars, test = novel cultivars.
    lodo    : leave-one-dataset-out; train = the two other datasets, test =
              every class of the held-out dataset.  Test classes are tagged
              'seen' (cultivar present in train under another domain) or
              'novel'.
    """
    m = meta[meta["drop"] == 0] if "drop" in meta else meta
    if protocol == "unified":
        folds = cultivar_folds(m, 3, seed)
        novel = set(folds[fold])
        test = m[m.cultivar.isin(novel)]
        train = m[~m.cultivar.isin(novel)]
        info = {"novel": sorted(novel), "base": sorted(set(train.cultivar))}
    elif protocol == "lodo":
        held = DATASETS[fold]
        test = m[m.dataset == held]
        train = m[m.dataset != held]
        seen = set(train.cultivar)
        info = {"held_out": held,
                "seen": sorted(set(test.cultivar) & seen),
                "novel": sorted(set(test.cultivar) - seen),
                "base": sorted(seen)}
    else:
        raise ValueError(protocol)
    return train.idx.to_numpy(), test.idx.to_numpy(), info


class PKSampler:
    """P classes x M images per batch, class-balanced (images drawn uniformly within a class)."""

    def __init__(self, labels, P, M, seed=0):
        self.rng = np.random.default_rng(seed)
        self.labels = np.asarray(labels)
        self.classes = np.unique(self.labels)
        self.by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.P, self.M = min(P, len(self.classes)), M

    def sample(self):
        cls = self.rng.choice(self.classes, self.P, replace=False)
        out = []
        for c in cls:
            pool = self.by_class[c]
            out.append(self.rng.choice(pool, self.M, replace=len(pool) < self.M))
        return np.concatenate(out)  # positions into labels array, class-contiguous


def sample_episode(rng, labels, groups, classes, n_way, k_shot, n_query, group_disjoint=True,
                   domains=None, support_domain=None, query_domain=None):
    """Sample an N-way K-shot episode; returns (support_pos, ys, query_pos, yq, classes).

    group_disjoint: support and query images never share a capture group, so a
    query can never be a near-copy / another view of a support fruit.
    """
    cls = rng.choice(classes, n_way, replace=False)
    sp, ys, qp, yq = [], [], [], []
    for i, c in enumerate(cls):
        pos = np.where(labels == c)[0]
        if support_domain is not None:
            s_pool = pos[domains[pos] == support_domain]
            q_pool = pos[domains[pos] == query_domain]
        else:
            s_pool = q_pool = pos
        if group_disjoint:
            # draw support groups at random until K images are collected,
            # query from the remaining groups
            g = groups[s_pool]
            ug = rng.permutation(np.unique(g))
            chosen, s = set(), []
            for gg in ug:
                members = s_pool[g == gg]
                s.extend(rng.permutation(members)[: k_shot - len(s)])
                chosen.add(gg)
                if len(s) >= k_shot:
                    break
            q_pool = q_pool[~np.isin(groups[q_pool], list(chosen))]
            s = np.array(s)
        else:
            s = rng.choice(s_pool, k_shot, replace=False)
            q_pool = np.setdiff1d(q_pool, s)
        q = rng.choice(q_pool, min(n_query, len(q_pool)), replace=False)
        sp.append(s); qp.append(q)
        ys += [i] * len(s); yq += [i] * len(q)
    return np.concatenate(sp), np.array(ys), np.concatenate(qp), np.array(yq), cls


def eligible_classes(labels, groups, k_shot, n_query, min_query=5):
    """Classes that can supply k_shot support + min_query group-disjoint queries."""
    out = []
    for c in np.unique(labels):
        _, cnt = np.unique(groups[labels == c], return_counts=True)
        cnt = np.sort(cnt)[::-1]
        # worst case: the support is drawn from the largest groups
        n_sup_groups = int(np.searchsorted(np.cumsum(cnt), k_shot)) + 1
        if cnt.sum() - cnt[:n_sup_groups].sum() >= min_query:
            out.append(c)
    return np.array(out)


def gpu_augment(x, out_res, scale=(0.45, 1.0), max_rot=25, jitter=0.15, train=True):
    """Per-sample random resized crop + flip + rotation + mild colour jitter.

    x: (B,3,S,S) float in [0,1] on GPU.  Hue is deliberately left untouched:
    skin colour is a genuine cultivar cue.
    """
    B = x.shape[0]
    dev = x.device
    if not train:
        return F.interpolate(x, size=(out_res, out_res), mode="bilinear",
                             antialias=True, align_corners=False)
    s = torch.empty(B, device=dev).uniform_(*scale).sqrt()          # side fraction
    ar = torch.exp(torch.empty(B, device=dev).uniform_(-0.2, 0.2))   # aspect jitter
    sx, sy = s * ar.sqrt(), s / ar.sqrt()
    sx, sy = sx.clamp(max=1), sy.clamp(max=1)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sx)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sy)
    ang = (torch.rand(B, device=dev) * 2 - 1) * max_rot * np.pi / 180
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([sx * cos * flip, -sy * sin, tx], 1),
        torch.stack([sx * sin * flip, sy * cos, ty], 1)], 1)
    # downscale first (antialias) so the sampled crop is not aliased
    if x.shape[-1] > out_res * 1.25:
        x = F.interpolate(x, size=(int(out_res * 1.25),) * 2, mode="bilinear",
                          antialias=True, align_corners=False)
    grid = F.affine_grid(theta, (B, 3, out_res, out_res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if jitter > 0:
        b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
        sat = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
        mean = x.mean((1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean
        x = x * b
        gray = x.mean(1, keepdim=True)
        x = (x - gray) * sat + gray
    return x.clamp(0, 1)
