"""Plot the learned per-dimension functions phi_j of the KAN metric.

usage: .venv/bin/python analysis/kan_curves.py <ckpt.pt> [<ckpt.pt> ...]
       .venv/bin/python analysis/kan_curves.py --paper <ckpt.pt> [<ckpt.pt> ...]
         (one panel per checkpoint, written to figures/fig_kan_metric_paper.png)

For each checkpoint that contains a KAN metric, the additive term
phi_j(u), u = |q_j - c_j| * sqrt(D), is evaluated on a grid.  We show
  (a) the median curve and the 10-90% band over all dimensions, next to the
      reference u^2 / D that squared Euclidean contributes per dimension,
  (b) the share of dimensions whose phi is increasing / decreasing /
      non-monotone / flat over the observed range (what the metric amplifies or
      suppresses).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import KANLinear  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BB = {"conv4": "Conv-4", "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
      "dinov2_s": "DINOv2 ViT-S/14", "dinov2_b": "DINOv2 ViT-B/14"}
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def phi_curves(sd, u):
    """Evaluate every phi_j on the grid u -> (D, len(u))."""
    w = {k[len("metric.kan."):]: v.float() for k, v in sd.items() if k.startswith("metric.kan.")}
    d = w["base_weight"].shape[1]
    layer = KANLinear(d, 1, grid_size=w["spline_weight"].shape[2] - 3, grid_range=(0.0, 4.0))
    layer.load_state_dict(w)
    with torch.no_grad():
        x = u[:, None].repeat(1, d)                       # (G, D)
        base = torch.nn.functional.silu(x) * layer.base_weight[0]
        sw = layer.spline_weight[0] * layer.spline_scaler[0][:, None]   # (D, K)
        spline = (layer.b_splines(x) * sw[None]).sum(-1)  # (G, D)
    return (base + spline).t().numpy(), d


def main(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = torch.linspace(0, 3.9, 200)
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        if not any(k.startswith("metric.kan.") for k in sd):
            print("no KAN metric in", p)
            continue
        phi, d = phi_curves(sd, u)
        uu = u.numpy()
        slope = np.diff(phi, axis=1)
        scale = max(1e-12, np.ptp(phi))
        tol = 1e-3 * scale / len(u)
        flat = np.ptp(phi, axis=1) < 1e-2 * scale
        inc = (slope >= -tol).all(1) & ~flat
        dec = (slope <= tol).all(1) & ~flat & ~inc
        inc, dec, flat = inc.mean(), dec.mean(), flat.mean()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        ax = axes[0]
        ax.fill_between(uu, np.percentile(phi, 10, 0), np.percentile(phi, 90, 0), color=BLUE, alpha=0.2,
                        lw=0, label="10–90% of dimensions")
        ax.plot(uu, np.median(phi, 0), color=BLUE, lw=2, label="median φⱼ")
        ax.plot(uu, uu ** 2 / d, color=ORANGE, lw=2, ls="--", label="Euclidean term u²/D")
        ax.set_xlabel("u = |qⱼ − cⱼ|·√D", fontsize=8, color=INK2)
        ax.set_ylabel("contribution to d(q, c)", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1]
        ax.bar(["increasing", "decreasing", "non-monotone", "flat"], [inc, dec, 1 - inc - dec - flat, flat],
               color=[BLUE, ORANGE, "#8a8984", GRID], width=0.6)
        ax.set_ylabel("share of dimensions", fontsize=8, color=INK2)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=8, colors=INK2)
            a.yaxis.grid(True, color=GRID, lw=0.6)
            a.set_axisbelow(True)
        name = os.path.splitext(os.path.basename(p))[0]
        fig.suptitle(f"Learned KAN metric – {name}", fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
        out = os.path.join(ROOT, "figures", f"fig_kan_metric_{name}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(out, f"increasing {inc:.2f} decreasing {dec:.2f} flat {flat:.2f}")


def paper(paths):
    """Compact figure for the paper: median phi and 10-90% band per checkpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = torch.linspace(0, 3.9, 200)
    uu = u.numpy()
    fig, axes = plt.subplots(1, len(paths), figsize=(3.0 * len(paths), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, paths):
        phi, d = phi_curves(torch.load(p, map_location="cpu"), u)
        ax.fill_between(uu, np.percentile(phi, 10, 0), np.percentile(phi, 90, 0), color=BLUE, alpha=0.2,
                        lw=0, label="10–90% of dimensions")
        ax.plot(uu, np.median(phi, 0), color=BLUE, lw=2, label="median φⱼ")
        ax.plot(uu, uu ** 2 / d, color=ORANGE, lw=2, ls="--", label="Euclidean term u²/D")
        proto, rest = os.path.basename(p).split("_", 1)
        bb = next(b for b in BB if rest.startswith(b + "_"))
        ax.set_title(f"{BB[bb]}, fold {int(proto[-1]) + 1}", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("u = |qⱼ − cⱼ|·√D", fontsize=8, color=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("contribution to d(q, c)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = os.path.join(ROOT, "figures", "fig_kan_metric_paper.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--paper"]:
        paper(sys.argv[2:])
    else:
        main(sys.argv[1:])
