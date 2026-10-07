"""'Normal' fine-tuning baselines and the closed-set limited-data study.

1. episode_finetune  - for novel-cultivar episodes: copy an ImageNet (or
   meta-trained) network, attach a fresh classifier and fine-tune the whole
   thing on the K-shot support set, then predict the queries.
2. closed_set_study  - all cultivars of one dataset are known; we train with
   K images per cultivar and test on held-out capture groups.  Compares
   ordinary fine-tuning against prototype classification with an embedding
   meta-trained on the *other* datasets (no labels of the target dataset are
   used except the K shots).
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import gpu_augment, sample_episode, eligible_classes
from evaluate import embed, macro_f1, ci95, proto_predict
from losses import CosineClassifier
from models import EmbeddingNet


def finetune_classifier(model, cache, train_idx, y, n_cls, device, steps, bs=32,
                        head_lr=1e-3, freeze_backbone=False):
    model = copy.deepcopy(model).to(device).train()
    clf = CosineClassifier(model.d_emb, n_cls).to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    groups = [{"params": list(model.head.parameters()) + list(clf.parameters()), "lr": head_lr}]
    if not freeze_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": model.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    y = np.asarray(y)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        pos = rng.choice(len(train_idx), bs, replace=len(train_idx) < bs)
        x = gpu_augment(cache.batch(train_idx[pos], device), model.res, train=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x)
        loss = F.cross_entropy(clf(z.float()), torch.as_tensor(y[pos], device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    model.eval()

    def predict(idx):
        Z, _ = embed(model, cache, idx, device)
        with torch.no_grad():
            return clf(Z.to(device)).softmax(1).cpu()
    return predict


def episode_finetune(cache, test_idx, base_model, device, shots=(1, 5), n_episodes=100, n_way=5,
                     n_query=15, steps=100, seed=0):
    """Full fine-tuning on each support set (the 'normal fine-tuning' arm)."""
    meta = cache.meta.set_index("idx").loc[test_idx]
    names = sorted(meta.cultivar.unique())
    labels = meta.cultivar.map({n: i for i, n in enumerate(names)}).to_numpy()
    groups = meta.group.to_numpy()
    out = {}
    for k in shots:
        classes = eligible_classes(labels, groups, k, n_query)
        rng = np.random.default_rng(seed * 1000 + k)
        accs, f1s, t0 = [], [], time.time()
        for e in range(n_episodes):
            sp, ys, qp, yq, _ = sample_episode(rng, labels, groups, classes, n_way, k, n_query)
            pred_fn = finetune_classifier(base_model, cache, test_idx[sp], ys, n_way, device, steps)
            pred = pred_fn(test_idx[qp]).argmax(1).numpy()
            accs.append(float((pred == yq).mean()))
            f1s.append(macro_f1(yq, pred, n_way))
        a, ac = ci95(accs)
        f, fc = ci95(f1s)
        out[f"finetune@{k}shot"] = {"acc": a, "acc_ci": ac, "f1": f, "f1_ci": fc,
                                    "n_episodes": n_episodes, "n_way": n_way,
                                    "sec_per_episode": (time.time() - t0) / n_episodes}
    return out


def group_holdout(meta, frac=0.3, seed=0, naive=False):
    """Per-cultivar split; whole capture groups go to test until ~frac."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c, m in meta.groupby("cultivar"):
        if naive:
            idx = rng.permutation(m.idx.to_numpy())
            n = int(round(frac * len(idx)))
            te += idx[:n].tolist(); tr += idx[n:].tolist()
            continue
        g = m.groupby("group").idx.apply(list)
        order = rng.permutation(len(g))
        n_te, target = 0, frac * len(m)
        for i in order:
            members = g.iloc[i]
            if n_te < target:
                te += members; n_te += len(members)
            else:
                tr += members
    return np.array(tr), np.array(te)


def pick_k(meta_tr, k, rng):
    """K training images per cultivar, spread over as many groups as possible."""
    out = []
    for c, m in meta_tr.groupby("cultivar"):
        if k == "all":
            out += m.idx.tolist(); continue
        firsts = m.sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("group")
        pick = firsts.idx.tolist()[:k]
        if len(pick) < k:
            rest = m[~m.idx.isin(pick)].idx.to_numpy()
            pick += rng.choice(rest, min(k - len(pick), len(rest)), replace=False).tolist()
        out += pick
    return np.array(out)


def score(probs, y, n):
    pred = probs.argmax(1).numpy()
    rec = [float((pred[y == c] == c).mean()) for c in range(n) if (y == c).any()]
    return {"acc": float((pred == y).mean()), "f1": macro_f1(y, pred, n), "bal_acc": float(np.mean(rec))}


def closed_set_study(cache, dataset, device, backbone, heads, ks, seeds, meta_ckpt=None,
                     naive=False, ft_steps=None):
    meta = cache.meta[(cache.meta.dataset == dataset) & (cache.meta["drop"] == 0)]
    names = sorted(meta.cultivar.unique())
    cmap = {n: i for i, n in enumerate(names)}
    tr_all, te = group_holdout(meta, seed=0, naive=naive)
    mi = cache.meta.set_index("idx")
    y_te = mi.loc[te].cultivar.map(cmap).to_numpy()
    rows = []
    for k in ks:
        for seed in (seeds if k != "all" else seeds[:1]):
            rng = np.random.default_rng(seed)
            tr = pick_k(mi.loc[tr_all].reset_index(), k, rng)
            y_tr = mi.loc[tr].cultivar.map(cmap).to_numpy()
            n_img = len(tr)
            steps = ft_steps or int(np.clip(20 * n_img / 32, 150, 1500))
            # (a) normal fine-tuning from ImageNet, one run per head
            for head in heads:
                torch.manual_seed(seed)
                m0 = EmbeddingNet(backbone, head, True)
                t0 = time.time()
                pf = finetune_classifier(m0, cache, tr, y_tr, len(names), device, steps)
                rows.append({"method": "finetune_imagenet", "head": head, "k": k, "seed": seed,
                             **score(pf(te), y_te, len(names)), "sec": time.time() - t0})
            # (b) prototypes on frozen ImageNet features (no training)
            m0 = EmbeddingNet(backbone, "none", True).to(device)
            rows.append({"method": "proto_imagenet", "head": "none", "k": k, "seed": seed,
                         **proto_rows(m0, cache, tr, y_tr, te, y_te, len(names), device)})
            # (c) prototypes / fine-tuning from the meta-trained embedding
            if meta_ckpt:
                for src_method, path in meta_ckpt.items():
                    sd = torch.load(path, map_location="cpu")
                    head = "kan" if any(k_.startswith("head.l1.") for k_ in sd) else "mlp"
                    metric = "kan" if any(k_.startswith("metric.") for k_ in sd) else "euclid"
                    mm = EmbeddingNet(backbone, head, False, metric=metric)
                    mm.load_state_dict({k_: v.float() if v.is_floating_point() else v for k_, v in sd.items()})
                    mm = mm.to(device)
                    rows.append({"method": f"proto_meta[{src_method}]", "head": head, "k": k, "seed": seed,
                                 **proto_rows(mm, cache, tr, y_tr, te, y_te, len(names), device)})
                    t0 = time.time()
                    pf = finetune_classifier(mm, cache, tr, y_tr, len(names), device, steps)
                    rows.append({"method": f"finetune_meta[{src_method}]", "head": head, "k": k, "seed": seed,
                                 **score(pf(te), y_te, len(names)), "sec": time.time() - t0})
            print(dataset, k, seed, rows[-1], flush=True)
    for r in rows:
        r.update({"dataset": dataset, "backbone": backbone, "naive_split": naive,
                  "n_train_pool": len(tr_all), "n_test": len(te)})
    return rows


@torch.no_grad()
def proto_rows(model, cache, tr, y_tr, te, y_te, n, device):
    Zs, _ = embed(model, cache, tr, device)
    Zq, _ = embed(model, cache, te, device)
    metric = copy.deepcopy(model.metric).cpu() if model.metric is not None else None
    _, p = proto_predict(Zs, torch.as_tensor(y_tr), Zq, n, metric)
    return score(p, y_te, n)
