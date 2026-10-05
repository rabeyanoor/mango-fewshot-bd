"""Episodic few-shot evaluation on frozen embeddings.

For every trained model we embed the test images once and then score
hundreds of N-way K-shot episodes with several inference rules:
  proto_z   nearest prototype on the (normalised) head output z
  proto_f   nearest prototype on the (normalised) backbone feature f
  proto_fc  as proto_f after centring on the base-class mean (SimpleShot CL2N)
  proto_fs  as proto_f after centring on the mean feature of the image's own source
            dataset, computed from the unlabelled test images of that source
            (transductive domain normalisation; no labels are used)
  logreg_z  logistic regression fitted on the support embeddings
The episode seeds depend only on (protocol, fold, K) so every model is scored
on identical episodes - paired comparisons are therefore valid.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import gpu_augment, sample_episode, eligible_classes


@torch.no_grad()
def embed(model, cache, idx, device, bs=128):
    model.eval()
    zs, fs = [], []
    for s in range(0, len(idx), bs):
        x = gpu_augment(cache.batch(idx[s:s + bs], device), model.res, train=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z, f = model(x, return_feat=True)
        zs.append(z.float().cpu()); fs.append(f.float().cpu())
    return torch.cat(zs), torch.cat(fs)


def macro_f1(y, p, n):
    f1 = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def ci95(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def proto_dist(es, ys, eq, n_way, metric=None):
    es, eq = F.normalize(es, dim=1), F.normalize(eq, dim=1)
    protos = torch.stack([es[ys == c].mean(0) for c in range(n_way)])
    if metric is None:
        return torch.cdist(eq, protos).pow(2)
    with torch.no_grad():
        return metric(eq, protos)


def proto_predict(es, ys, eq, n_way, metric=None):
    """Nearest prototype.  Confidences use a fixed softmax temperature (10) for
    every model, so ECE compares raw distance scales, not calibrated models."""
    d = proto_dist(es, ys, eq, n_way, metric)
    return d.argmin(1), torch.softmax(-10 * d, 1)


def logreg_predict(es, ys, eq, n_way, steps=100, wd=1e-3):
    es, eq = F.normalize(es, dim=1), F.normalize(eq, dim=1)
    W = torch.zeros(es.shape[1], n_way, requires_grad=True)
    b = torch.zeros(n_way, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    yt = torch.as_tensor(ys)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(10 * es @ W + b, yt) + wd * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = torch.softmax(10 * eq @ W + b, 1)
    return p.argmax(1), p


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def run_episodes(Z, Fe, labels, groups, class_names, *, n_way=5, shots=(1, 5, 10), n_query=15,
                 n_episodes=600, seed=0, rules=("proto_z", "proto_f", "logreg_z"),
                 domains=None, domain_pair=None, metric=None, base_mean=None, src_centred=None):
    """Z, Fe: (N, d) tensors aligned with labels/groups (numpy).  Rules are
    '<classifier>_<features>' with features z (head output), f (backbone) or
    fc (backbone centred on base_mean).  Returns a dict."""
    feats = {"z": Z, "f": Fe}
    if base_mean is not None:
        feats["fc"] = F.normalize(Fe, dim=1) - base_mean
    if src_centred is not None:
        feats["fs"] = src_centred
    out = {}
    labels, groups = np.asarray(labels), np.asarray(groups)
    for k in shots:
        if domain_pair is None:
            classes = eligible_classes(labels, groups, k, n_query)
        else:
            a, b = domain_pair
            classes = np.array([c for c in np.unique(labels)
                                if ((labels == c) & (domains == a)).sum() >= k
                                and ((labels == c) & (domains == b)).sum() >= 5])
        nw = min(n_way, len(classes))
        if nw < 2:
            continue
        rng = np.random.default_rng(seed * 1000 + k)
        rec = {r: {"acc": [], "f1": [], "conf": [], "correct": []} for r in rules}
        hits = {r: np.zeros((len(class_names), len(class_names)), dtype=np.int64) for r in rules}
        for _ in range(n_episodes):
            sp, ys, qp, yq, cls = sample_episode(
                rng, labels, groups, classes, nw, k, n_query,
                group_disjoint=domain_pair is None, domains=domains,
                support_domain=None if domain_pair is None else domain_pair[0],
                query_domain=None if domain_pair is None else domain_pair[1])
            ys_t = torch.as_tensor(ys)
            for r in rules:
                E = feats[r.split("_", 1)[1]]
                if r.startswith("proto"):
                    pred, prob = proto_predict(E[sp], ys_t, E[qp], nw,
                                               metric if r == "proto_z" else None)
                else:
                    pred, prob = logreg_predict(E[sp], ys_t, E[qp], nw)
                pred = pred.numpy()
                rec[r]["acc"].append(float((pred == yq).mean()))
                rec[r]["f1"].append(macro_f1(yq, pred, nw))
                rec[r]["conf"] += prob.max(1).values.tolist()
                rec[r]["correct"] += (pred == yq).tolist()
                np.add.at(hits[r], (cls[yq], cls[pred]), 1)
        for r in rules:
            acc, acc_ci = ci95(rec[r]["acc"])
            f1, f1_ci = ci95(rec[r]["f1"])
            conf = hits[r]
            recall = {class_names[i]: float(conf[i, i] / conf[i].sum())
                      for i in range(len(class_names)) if conf[i].sum() > 0}
            out[f"{r}@{k}shot"] = {
                "acc": acc, "acc_ci": acc_ci, "f1": f1, "f1_ci": f1_ci,
                "bal_acc": float(np.mean(list(recall.values()))),
                "ece": ece(rec[r]["conf"], rec[r]["correct"]),
                "n_way": nw, "n_episodes": n_episodes,
                "recall": recall,
                "confusion": conf.tolist() if r == "proto_z" else None,
                # per-episode accuracies -> paired significance tests between
                # models (episodes are identical across models by seeding)
                "episode_acc": [round(a, 4) for a in rec[r]["acc"]] if r.startswith("proto") else None,
            }
    return out


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def open_set_eval(Z, labels, groups, n_way=5, shots=(1, 5), n_query=15, n_episodes=300,
                  seed=0, metric=None, name="open"):
    """Open-set episodes: N known cultivars + 1 unknown cultivar whose queries
    must be rejected.  Score = distance to the nearest prototype; we report
    AUROC (unknown vs known queries) and closed-set accuracy on the knowns."""
    out = {}
    labels, groups = np.asarray(labels), np.asarray(groups)
    for k in shots:
        classes = eligible_classes(labels, groups, k, n_query)
        nw = min(n_way, len(classes) - 1)
        if nw < 2:
            continue
        rng = np.random.default_rng(seed * 1000 + k)
        aucs, accs = [], []
        for _ in range(n_episodes):
            sp, ys, qp, yq, _ = sample_episode(rng, labels, groups, classes, nw + 1, k, n_query)
            known_s = ys < nw
            d = proto_dist(Z[sp[known_s]], torch.as_tensor(ys[known_s]), Z[qp], nw, metric)
            score = d.min(1).values.numpy()
            unk = yq == nw
            aucs.append(auroc(score[unk], score[~unk]))
            accs.append(float((d.argmin(1).numpy()[~unk] == yq[~unk]).mean()))
        a, ac = ci95(aucs)
        b, bc = ci95(accs)
        out[f"{name}@{k}shot"] = {"auroc": a, "auroc_ci": ac, "closed_acc": b, "closed_acc_ci": bc,
                                "n_way": nw, "n_episodes": n_episodes,
                                "episode_auroc": [round(x, 4) for x in aucs]}
    return out
