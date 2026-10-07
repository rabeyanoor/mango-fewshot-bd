"""Turn the Kaggle result files into paper tables and figures.

usage: .venv/bin/python analysis/aggregate.py [results_dir] [out_dir]

Reads every results*.jsonl under results_dir and writes
  <out>/tables.md          all tables (markdown, mean +- 95% CI)
  <out>/summary.json       machine-readable version of the main numbers
  figures/*.png            figures used in the paper
Fold aggregation: mean of the three fold means; the CI half-width of the
mean of independent folds is sqrt(sum ci_f^2) / n_folds.
Paired tests: Wilcoxon signed-rank over identical episodes (all folds
pooled), Holm-corrected within each table.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "kaggle")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BB_ORDER = ["conv4", "resnet18", "resnet50", "densenet121", "dinov2_s", "dinov2_b"]
BB_NAME = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2",
           "fusion": "Fusion (5 backbones)", "conv4": "Conv-4", "resnet18": "ResNet-18", "resnet50": "ResNet-50",
           "densenet121": "DenseNet-121", "dinov2_s": "DINOv2 ViT-S/14", "dinov2_b": "DINOv2 ViT-B/14"}
M_ORDER = ["imagenet", "ce", "proto", "proto_mlp", "proto_kanhead", "proto_margin", "kan_metric", "ours",
           "arcface", "supcon", "triplet", "proto_tri", "ours_kanhead"]
M_NAME = {"imagenet": "ImageNet features (no training)", "ce": "Fine-tuned CE (cosine)",
          "proto": "ProtoNet", "proto_mlp": "ProtoNet + MLP head", "proto_kanhead": "ProtoNet + KAN head",
          "proto_margin": "+ prototype margin", "kan_metric": "+ KAN metric",
          "ours": "KAN metric + margin (ours)", "arcface": "ArcFace", "supcon": "SupCon",
          "triplet": "Triplet (batch-hard)", "proto_tri": "ProtoNet + triplet",
          "ours_kanhead": "Ours + KAN head"}
# fixed categorical slots (reference palette, light mode); colour follows the method
COLOR = {"ours": "#2a78d6", "proto_mlp": "#eb6834", "kan_metric": "#1baf7a", "proto_margin": "#eda100",
         "proto_kanhead": "#e87ba4", "proto": "#008300", "ce": "#4a3aa7", "imagenet": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(res_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(res_dir, "**", "results*.jsonl"), recursive=True)):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    # keep the last copy of each key (re-runs overwrite)
    return list({r["key"]: r for r in rows}.values())


def flat(records):
    """One row per (record, section, rule@shot)."""
    out = []
    for r in records:
        if r.get("kind") not in ("train", "zeroshot", "reeval", "fusion"):
            continue
        c = {**r.get("cfg", {}), **r["cfg_in"]}
        if c.get("n_episodes", 600) != 600:  # smoke / debug runs
            continue
        method = "imagenet" if r["kind"] == "zeroshot" else r.get("method")
        base = dict(method=method, backbone=c.get("backbone"), protocol=c.get("protocol"),
                    fold=c.get("fold"), variant=c.get("variant", "crop"), seed=c.get("seed", 0),
                    margin=c.get("margin", 0.0), mining=c.get("mining", "batch_hard"),
                    pretrained=c.get("pretrained", True), ev=c.get("eval", 1))
        for sec, d in r["results"].items():
            for rk, v in d.items():
                rule, shot = rk.split("@")
                out.append({**base, "section": sec, "rule": rule, "shot": shot, **{
                    k: v.get(k) for k in ("acc", "acc_ci", "f1", "f1_ci", "bal_acc", "ece",
                                          "auroc", "auroc_ci", "closed_acc", "n_way")},
                    "episode_acc": v.get("episode_acc"), "episode_auroc": v.get("episode_auroc"),
                    "recall": v.get("recall"), "confusion": v.get("confusion")})
    df = pd.DataFrame(out)
    if df.empty or "ev" not in df or (df.ev == 1).all():
        return df
    # Second evaluation pass (eval=2: saved models re-scored with extra rules on the
    # same episodes).  Numbers of the original evaluation are kept; the re-evaluation
    # only adds rules/sections and per-episode accuracies that the first pass lacked.
    ident = ["method", "backbone", "protocol", "fold", "variant", "seed", "margin", "mining",
             "pretrained", "section", "rule", "shot"]
    df = df.sort_values("ev", kind="stable")
    return df.groupby(ident, dropna=False, sort=False).first().reset_index()


def default_cfg(df):
    """Rows of the main grid: crop images, seed 0, default margin/mining."""
    m = (df.variant == "crop") & (df.seed == 0) & (df.mining == "batch_hard") & (df.pretrained == True)  # noqa
    dm = df.method.map({"ours": 0.1, "proto_margin": 0.1, "ours_kanhead": 0.1}).fillna(0.0)
    return df[m & (np.isclose(df.margin.astype(float), dm))]


def fold_agg(g, col="acc", ci="acc_ci"):
    vals = g[col].astype(float).to_numpy()
    cis = g[ci].astype(float).to_numpy() if ci in g else np.zeros_like(vals)
    return vals.mean(), np.sqrt((cis ** 2).sum()) / len(vals), len(vals)


def fmt(m, h, n=None, n_req=3):
    if np.isnan(m):
        return "–"
    s = f"{100 * m:.2f} ± {100 * h:.2f}"
    return s + ("" if n is None or n >= n_req else f" ({n}f)")


def wilcoxon_holm(pairs):
    """pairs: {label: (a_list, b_list)} -> {label: (p_holm, mean_diff)}"""
    from scipy.stats import wilcoxon
    raw = {}
    for k, (a, b) in pairs.items():
        a, b = np.asarray(a), np.asarray(b)
        if len(a) != len(b) or len(a) < 10 or np.allclose(a, b):
            raw[k] = (1.0, float((a - b).mean()) if len(a) == len(b) else np.nan)
            continue
        raw[k] = (float(wilcoxon(a, b, zero_method="zsplit").pvalue), float((a - b).mean()))
    order = sorted(raw, key=lambda k: raw[k][0])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * raw[k][0]))
        out[k] = (running, raw[k][1])
    return out


def episodes(df, method, bb, shot, section="all", rule="proto_z", protocol="unified", key="episode_acc"):
    g = df[(df.method == method) & (df.backbone == bb) & (df.shot == shot) & (df.section == section)
           & (df.rule == rule) & (df.protocol == protocol)].sort_values("fold")
    if g.empty or g[key].isna().any():
        return None
    return sum((list(x) for x in g[key]), [])


def table_main(df, lines, summary, rule="proto_z"):
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & (d.section == "all") & (d.rule == rule)]
    if d.empty:
        return
    lines.append(f"\n## Table 1 – Novel-cultivar 5-way accuracy (%), unified benchmark, rule `{rule}`\n")
    lines.append("Mean over 3 cultivar folds × 600 episodes, ± 95% CI. "
                 "Bold = best per backbone and shot. † = significantly different from *ours* "
                 "(Wilcoxon, Holm, p<0.05).\n")
    shots = ["1shot", "5shot", "10shot"]
    present = [b for b in BB_ORDER if not d[d.backbone == b].empty]
    lines.append("| Method | " + " | ".join(f"{BB_NAME[b]} {s[:-4]}-shot" for b in present for s in shots) + " |")
    lines.append("|" + "---|" * (1 + len(present) * len(shots)))
    best, cells = {}, {}
    for b in present:
        for s in shots:
            tests = {}
            ref = episodes(d, "ours", b, s, rule=rule)
            for meth in M_ORDER:
                g = d[(d.method == meth) & (d.backbone == b) & (d.shot == s)]
                if g.empty:
                    continue
                cells[(meth, b, s)] = fold_agg(g)
                other = episodes(d, meth, b, s, rule=rule)
                if ref is not None and other is not None and meth != "ours":
                    tests[meth] = (ref, other)
            for meth, (p, _) in wilcoxon_holm(tests).items():
                cells[(meth, b, s)] = cells[(meth, b, s)] + (p,)
            vals = {k: v[0] for k, v in cells.items() if k[1] == b and k[2] == s}
            if vals:
                best[(b, s)] = max(vals, key=vals.get)
    for meth in M_ORDER:
        if not any(k[0] == meth for k in cells):
            continue
        row = [M_NAME[meth]]
        for b in present:
            for s in shots:
                v = cells.get((meth, b, s))
                if v is None:
                    row.append("–"); continue
                t = fmt(v[0], v[1], v[2])
                if len(v) > 3 and v[3] < 0.05:
                    t += " †"
                if best.get((b, s)) == (meth, b, s):
                    t = f"**{t}**"
                row.append(t)
                summary.setdefault("main", {}).setdefault(meth, {})[f"{b}@{s}"] = round(100 * v[0], 2)
        lines.append("| " + " | ".join(row) + " |")


def table_rule(df, lines):
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & (d.section == "all") & (d.shot == "5shot")]
    if d.empty:
        return
    lines.append("\n## Table 2 – Inference rule on the same embeddings (5-shot, %)\n")
    lines.append("`proto_z`: nearest prototype with the model's metric on the head output; `proto_f`: Euclidean "
                 "prototype on backbone features; `proto_fc`: the same after base-mean centring (CL2N); "
                 "`logreg_z`: logistic regression fitted on the support set.\n")
    lines.append("| Backbone | Method | proto_z | proto_f | proto_fc | logreg_z |")
    lines.append("|---|---|---|---|---|---|")
    for b in BB_ORDER:
        for meth in M_ORDER:
            g = d[(d.backbone == b) & (d.method == meth)]
            if g.empty:
                continue
            row = [BB_NAME[b], M_NAME[meth]]
            for r in ("proto_z", "proto_f", "proto_fc", "logreg_z"):
                gg = g[g.rule == r]
                row.append(fmt(*fold_agg(gg)) if not gg.empty else "–")
            lines.append("| " + " | ".join(row) + " |")


def table_lodo(df, lines):
    d = default_cfg(df)
    d = d[(d.protocol == "lodo") & (d.rule == "proto_z") & (d.shot == "5shot")
          & d.section.isin(["all", "seen", "novel"])]
    if d.empty:
        return
    lines.append("\n## Table 4 – Leave-one-dataset-out, 5-way 5-shot accuracy (%)\n")
    lines.append("Train on two datasets, test on every cultivar of the third. *seen* = cultivar also present "
                 "in training (pure domain shift); *novel* = unseen cultivar and unseen domain.\n")
    held = {0: "MangoImageBD", 1: "MangoClassify12", 2: "Mangifera2012"}
    lines.append("| Backbone | Method | " + " | ".join(f"{held[f]} {s}" for f in range(3)
                                                       for s in ("all", "seen", "novel")) + " |")
    lines.append("|---|---|" + "---|" * 9)
    for b in BB_ORDER:
        for meth in M_ORDER:
            g = d[(d.backbone == b) & (d.method == meth)]
            if g.empty:
                continue
            row = [BB_NAME[b], M_NAME[meth]]
            for f in range(3):
                for s in ("all", "seen", "novel"):
                    gg = g[(g.fold == f) & (g.section == s)]
                    row.append(fmt(gg.acc.iat[0], gg.acc_ci.iat[0]) if not gg.empty else "–")
            lines.append("| " + " | ".join(row) + " |")


def table_crossdomain(df, lines):
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & d.section.str.startswith("x:") & (d.rule == "proto_z")]
    if d.empty:
        return
    lines.append("\n## Table 5 – Cross-domain episodes (support from dataset A, query from dataset B), %\n")
    lines.append("Averaged over all ordered dataset pairs and folds where ≥2 novel cultivars are shared; "
                 "compare with the in-domain numbers of Table 1.\n")
    lines.append("| Backbone | Method | 1-shot | 5-shot |")
    lines.append("|---|---|---|---|")
    for b in BB_ORDER:
        for meth in M_ORDER:
            g = d[(d.backbone == b) & (d.method == meth)]
            if g.empty:
                continue
            row = [BB_NAME[b], M_NAME[meth]]
            for s in ("1shot", "5shot"):
                gg = g[g.shot == s]
                row.append(f"{100 * gg.acc.mean():.2f}" if not gg.empty else "–")
            lines.append("| " + " | ".join(row) + " |")


def table_open(df, lines):
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & (d.section == "open_set")]
    if d.empty:
        return
    lines.append("\n## Table 6 – Open-set episodes: AUROC (%) for rejecting an unseen cultivar\n")
    lines.append("| Backbone | Method | 1-shot AUROC | 5-shot AUROC | 5-shot closed acc |")
    lines.append("|---|---|---|---|---|")
    for b in BB_ORDER:
        for meth in M_ORDER:
            g = d[(d.backbone == b) & (d.method == meth)]
            if g.empty:
                continue
            row = [BB_NAME[b], M_NAME[meth]]
            for s in ("1shot", "5shot"):
                gg = g[g.shot == s]
                row.append(fmt(*fold_agg(gg, "auroc", "auroc_ci")[:2]) if not gg.empty else "–")
            gg = g[g.shot == "5shot"]
            row.append(f"{100 * gg.closed_acc.mean():.2f}" if not gg.empty else "–")
            lines.append("| " + " | ".join(row) + " |")


def table_followup(df, lines):
    """Background removal (seg), lightweight CNNs and feature fusion; prototypes on f."""
    d = df[(df.protocol == "unified") & (df.seed == 0) & (df.pretrained == True)]  # noqa: E712
    if not ((d.variant == "seg").any() or (d.backbone == "fusion").any()):
        return
    lines.append("\n## Table 12 – Background removal, lightweight CNNs and feature fusion, %\n")
    lines.append("Nearest prototype on the backbone feature f; LR = logistic regression on z; cross = "
                 "5-shot cross-domain episodes (as Table 5); AUROC = 5-shot open set.\n")
    lines.append("| Backbone | Model | Images | 1-shot | 5-shot | 10-shot | 5-shot LR | cross | AUROC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def cell(g, sec, rule, shot, col="acc"):
        g = g[(g.section.str.startswith("x:") if sec == "cross" else g.section == sec)
              & (g.rule == rule) & (g.shot == shot)]
        return f"{100 * g[col].mean():.1f}" if g.fold.nunique() == 3 else "–"
    for b in BB_ORDER[1:] + ["efficientnet_b0", "mobilenet_v2", "fusion"]:
        for meth in sorted(d[d.backbone == b].method.dropna().unique()):
            for v in ("crop", "seg"):
                g = d[(d.backbone == b) & (d.method == meth) & (d.variant == v)]
                if b != "fusion" and not (d[(d.backbone == b)].variant == "seg").any() and meth != "ce":
                    continue
                if b not in ("fusion", "efficientnet_b0", "mobilenet_v2") and meth != "imagenet" \
                        and not (b == "dinov2_b" and meth == "ce" and v == "crop"):
                    continue
                cells = [cell(g, "all", "proto_f", s) for s in ("1shot", "5shot", "10shot")]
                cells += [cell(g, "all", "logreg_z", "5shot"), cell(g, "cross", "proto_f", "5shot"),
                          cell(g, "open_set", "open_f", "5shot", "auroc")]
                if cells[1] == "–":
                    continue
                name = M_NAME.get(meth, meth.replace("fusion_", "fusion of 5 backbones, "))
                lines.append(f"| {BB_NAME[b]} | {name} | {'background removed' if v == 'seg' else 'crop'} | "
                             + " | ".join(cells) + " |")


def table_variants(df, lines):
    """Margin sensitivity, mining policies, background, seeds."""
    d = df[(df.protocol == "unified") & (df.section == "all") & (df.rule == "proto_z")]
    if d.empty:
        return
    lines.append("\n## Table 7 – Sensitivity and ablations (ResNet-18 unless stated), 1/5-shot %\n")
    lines.append("| Setting | 1-shot | 5-shot |")
    lines.append("|---|---|---|")

    def add(label, g):
        if g.empty:
            return
        cells = []
        for s in ("1shot", "5shot"):
            gg = g[g.shot == s]
            cells.append(fmt(*fold_agg(gg)) if not gg.empty else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    base = d[(d.variant == "crop") & (d.seed == 0) & (d.pretrained == True)]  # noqa
    for mg in (0.0, 0.05, 0.1, 0.2, 0.4):
        meth = "kan_metric" if mg == 0 else "ours"
        add(f"KAN metric, margin m={mg}", base[(base.backbone == "resnet18") & (base.method == meth)
                                              & np.isclose(base.margin.astype(float), mg)])
    for mn in ("random", "semi_hard", "batch_hard", "distance_weighted"):
        add(f"Triplet, {mn} mining", base[(base.backbone == "resnet18") & (base.method == "triplet")
                                          & (base.mining == mn)])
    for b in ("resnet18", "densenet121"):
        for meth in ("proto_mlp", "ours"):
            for v in ("crop", "full"):
                add(f"{BB_NAME[b]}, {M_NAME[meth]}, {'fruit crop' if v == 'crop' else 'full frame (background kept)'}",
                    d[(d.backbone == b) & (d.method == meth) & (d.variant == v) & (d.seed == 0)
                      & np.isclose(d.margin.astype(float), 0.1 if meth == "ours" else 0.0)])
    # seed variance
    for b in ("resnet18", "densenet121"):
        for meth in ("proto_mlp", "ours"):
            g = d[(d.backbone == b) & (d.method == meth) & (d.variant == "crop") & (d.shot == "5shot")
                  & np.isclose(d.margin.astype(float), 0.1 if meth == "ours" else 0.0)]
            per_seed = g.groupby("seed").acc.mean()
            if len(per_seed) > 1:
                lines.append(f"| {BB_NAME[b]}, {M_NAME[meth]}: 5-shot over seeds {list(per_seed.index)} | – | "
                             f"{100 * per_seed.mean():.2f} ± {100 * per_seed.std(ddof=1):.2f} (sd) |")


def table_closed(records, lines, summary):
    rows = [dict(r_, **{}) for r in records if r.get("kind") == "closed" for r_ in r["rows"]]
    if not rows:
        return None
    c = pd.DataFrame(rows)
    c["k"] = c.k.astype(str)
    g = c.groupby(["dataset", "backbone", "naive_split", "method", "head", "k"]).agg(
        acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean"), n=("acc", "size")).reset_index()
    lines.append("\n## Table 8 – Closed-set limited data: normal fine-tuning vs few-shot prototypes (macro-F1 %)\n")
    lines.append("All cultivars known; K training images per cultivar; test = held-out capture groups. "
                 "Mean over 3 draws of the K images.  `proto_meta[...]` uses an embedding meta-trained on the "
                 "*other two* datasets (LODO checkpoint) and only the K shots of the target dataset.\n")
    ks = ["1", "5", "10", "20", "all"]
    for (ds, bb), gg in g[~g.naive_split].groupby(["dataset", "backbone"]):
        lines.append(f"\n**{ds} – {BB_NAME.get(bb, bb)}**\n")
        lines.append("| Method | head | " + " | ".join(f"K={k}" for k in ks) + " |")
        lines.append("|---|---|" + "---|" * len(ks))
        for (meth, head), h in gg.groupby(["method", "head"]):
            vals = []
            for k in ks:
                v = h[h.k == k]
                vals.append(f"{100 * v.f1.iat[0]:.1f}" if not v.empty else "–")
            lines.append(f"| {meth} | {head} | " + " | ".join(vals) + " |")
    # leakage inflation
    lk = g[(g.k == "all") & (g.method == "finetune_imagenet")]
    if lk.naive_split.any():
        lines.append("\n## Table 9 – Leakage inflation: random image split vs capture-group split (K=all, fine-tuned, acc %)\n")
        lines.append("| Dataset | Backbone | head | group split | random split | inflation |")
        lines.append("|---|---|---|---|---|---|")
        for (ds, bb, head), h in lk.groupby(["dataset", "backbone", "head"]):
            a = h[~h.naive_split].acc
            b = h[h.naive_split].acc
            if len(a) and len(b):
                lines.append(f"| {ds} | {BB_NAME.get(bb, bb)} | {head} | {100 * a.iat[0]:.2f} | "
                             f"{100 * b.iat[0]:.2f} | {100 * (b.iat[0] - a.iat[0]):+.2f} |")
                summary.setdefault("leakage", {})[f"{ds}/{bb}/{head}"] = round(100 * (b.iat[0] - a.iat[0]), 2)
    return c


def table_episode_ft(records, df, lines):
    rows = [r for r in records if r.get("kind") == "episode_ft"]
    if not rows:
        return
    lines.append("\n## Table 10 – Per-episode full fine-tuning (\"normal fine-tuning\") on novel cultivars, %\n")
    lines.append("ImageNet backbone + fresh classifier fine-tuned on each support set for 100 steps "
                 "(100 episodes per fold).  Compare with prototype inference in Table 1.\n")
    lines.append("| Backbone | fold | 1-shot | 5-shot | s / episode |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        c = r["cfg_in"]
        res = r["results"]
        lines.append(f"| {BB_NAME[c['backbone']]} | {c['fold']} | "
                     f"{fmt(res['finetune@1shot']['acc'], res['finetune@1shot']['acc_ci'])} | "
                     f"{fmt(res['finetune@5shot']['acc'], res['finetune@5shot']['acc_ci'])} | "
                     f"{res['finetune@5shot']['sec_per_episode']:.1f} |")


def table_efficiency(records, lines):
    rows = [r_ for r in records if r.get("kind") == "efficiency" for r_ in r["rows"]]
    if not rows:
        return
    e = pd.DataFrame(rows)
    lines.append("\n## Table 11 – Cost (T4 GPU fp16; CPU 4 threads)\n")
    cols = [c for c in ["backbone", "head", "metric", "params_backbone_M", "params_head_M", "params_metric_M",
                        "gmacs", "gpu_ms_b1", "gpu_ms_b32", "cpu_ms_b1", "gpu_peak_mem_mb_b32",
                        "metric_ms_75x5"] if c in e]
    lines.append(e[cols].round(3).to_markdown(index=False))


def table_dataset(lines, meta_path, group_paths=()):
    m = pd.read_csv(meta_path)
    if group_paths:
        m = m.merge(pd.read_csv(sorted(group_paths, key=os.path.getmtime)[-1]), on="idx")
        m = m[m["drop"] == 0]
    lines.append("\n## Table 0 – The unified MangoFS-BD benchmark (images / capture groups)\n")
    pv = m.pivot_table(index="cultivar", columns="dataset", values="idx", aggfunc="count", fill_value=0)
    if "group" in m:
        gv = m.pivot_table(index="cultivar", columns="dataset", values="group", aggfunc="nunique", fill_value=0)
        pv = pv.astype(str) + " / " + gv.astype(str)
    lines.append(pv.to_markdown())


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_main(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & (d.section == "all") & (d.rule == "proto_z")]
    if d.empty:
        return
    meths = [m for m in M_ORDER if m in COLOR and m in set(d.method)]
    bbs = [b for b in BB_ORDER if b in set(d.backbone)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, s in zip(axes, ("1shot", "5shot")):
        w = 0.8 / len(meths)
        for i, meth in enumerate(meths):
            xs, ys, es = [], [], []
            for j, b in enumerate(bbs):
                g = d[(d.method == meth) & (d.backbone == b) & (d.shot == s)]
                if g.empty:
                    continue
                m_, h_, _ = fold_agg(g)
                xs.append(j - 0.4 + w * (i + 0.5)); ys.append(100 * m_); es.append(100 * h_)
            ax.bar(xs, ys, w * 0.9, color=COLOR[meth], label=M_NAME[meth], edgecolor="white", linewidth=0.5)
            ax.errorbar(xs, ys, es, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
        ax.set_xticks(range(len(bbs)))
        ax.set_xticklabels([BB_NAME[b].replace(" ViT-S/14", "-S") for b in bbs], fontsize=8)
        ax.set_title(f"5-way {s[:-4]}-shot, novel cultivars", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("accuracy (%)", fontsize=8, color=INK2)
        hs = [p.get_height() for p in ax.patches] or [0, 100]
        ax.set_ylim(max(0, min(hs) - 5), min(100, max(hs) + 3))
        _style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_main_accuracy.png"), dpi=200)
    plt.close(fig)


def fig_closed(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if c is None:
        return
    c = c[~c.naive_split]
    ks = ["1", "5", "10", "20", "all"]
    for bb, cb in c.groupby("backbone"):
        dss = sorted(cb.dataset.unique())
        fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        series = [("finetune_imagenet", "mlp", "Fine-tune ImageNet (MLP head)", COLOR["ce"]),
                  ("finetune_imagenet", "kan", "Fine-tune ImageNet (KAN head)", COLOR["proto_kanhead"]),
                  ("proto_imagenet", "none", "Prototypes, ImageNet features", COLOR["imagenet"]),
                  ("proto_meta[proto_mlp]", "mlp", "Prototypes, LODO ProtoNet + MLP", COLOR["proto_mlp"]),
                  ("proto_meta[ours]", "mlp", "Prototypes, LODO KAN-Proto", COLOR["ours"]),
                  ("finetune_meta[proto_mlp]", "mlp", "Fine-tune from LODO ProtoNet + MLP", COLOR["proto_margin"]),
                  ("finetune_meta[ours]", "mlp", "Fine-tune from LODO KAN-Proto", COLOR["kan_metric"])]
        for ax, ds in zip(axes, dss):
            g = cb[cb.dataset == ds]
            for meth, head, label, col in series:
                h = g[(g.method == meth) & (g["head"] == head)].groupby("k").f1.mean()
                xs = [i for i, k in enumerate(ks) if k in h.index]
                if not xs:
                    continue
                ax.plot(xs, [100 * h[ks[i]] for i in xs], marker="o", ms=4, lw=2, color=col, label=label)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"K={k}" for k in ks], fontsize=8)
            ax.set_title(ds, fontsize=9, color=INK, loc="left")
            _style(ax)
        axes[0].set_ylabel("macro-F1 (%)", fontsize=8, color=INK2)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, fontsize=7, frameon=False, loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(os.path.join(FIG, f"fig_finetune_vs_fewshot_{bb}.png"), dpi=200)
        plt.close(fig)


def fig_confusion(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = default_cfg(df)
    d = d[(d.protocol == "unified") & (d.section == "all") & (d.rule == "proto_z") & (d.shot == "5shot")
          & (d.method == "proto")]
    if d.empty:
        return
    b = max(d.backbone.unique(), key=lambda x: d[d.backbone == x].acc.mean())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (_, r) in zip(axes, d[d.backbone == b].sort_values("fold").iterrows()):
        names = sorted(r.recall.keys())
        conf = np.array(r.confusion, dtype=float)
        nz = conf.sum(1) > 0
        conf = conf[nz][:, nz]
        conf = conf / conf.sum(1, keepdims=True)
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                if conf[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * conf[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if conf[i, j] > 0.6 else INK)
        ax.set_title(f"fold {r.fold + 1}", fontsize=9, loc="left", color=INK)
    fig.suptitle(f"ProtoNet, {BB_NAME[b]}, 5-shot: row-normalised confusion (%) over 600 episodes",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=200)
    plt.close(fig)


def main():
    records = load(RES)
    print(len(records), "records")
    df = flat(records)
    lines = ["# MangoFS-BD – results\n", f"_{len(records)} result records from `{os.path.relpath(RES, ROOT)}`._\n"]
    summary = {}
    bench = os.path.join(ROOT, "data", "benchmark")
    if not os.path.exists(os.path.join(bench, "meta.csv")):
        bench = os.path.join(ROOT, "benchmark")   # metadata copy shipped with the repository
    if os.path.exists(os.path.join(bench, "meta.csv")):
        # the benchmark files used by every experiment (not copies left in old kernel outputs)
        table_dataset(lines, os.path.join(bench, "meta.csv"), [os.path.join(bench, "groups.csv")])
    if not df.empty:
        table_main(df, lines, summary)
        table_rule(df, lines)
        table_lodo(df, lines)
        table_crossdomain(df, lines)
        table_open(df, lines)
        table_variants(df, lines)
        fig_main(df)
        fig_confusion(df)
    c = table_closed(records, lines, summary)
    fig_closed(c)
    table_episode_ft(records, df, lines)
    table_efficiency(records, lines)
    if not df.empty:
        table_followup(df, lines)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", os.path.join(OUT, "tables.md"))


if __name__ == "__main__":
    main()
