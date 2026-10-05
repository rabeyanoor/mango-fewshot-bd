"""Deployment cost of every backbone x head (x metric) combination.

Reports parameters, forward GFLOPs at the model's input resolution, GPU
latency (batch 1 and batch 32) and CPU latency (batch 1, 4 threads), plus
the extra cost of scoring one 5-way query with the KAN metric.
"""
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from models import EmbeddingNet, count_params

COMBOS = [(bb, h, mt) for bb in ["conv4", "resnet18", "resnet50", "densenet121", "dinov2_s", "dinov2_b"]
          for h, mt in [("none", "euclid"), ("mlp", "euclid"), ("kan", "euclid"), ("mlp", "kan")]]


def _latency(fn, n_warm=5, n_iter=30, sync=None):
    for _ in range(n_warm):
        fn()
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / n_iter * 1000


@torch.no_grad()
def measure(bb, head, metric, device):
    m = EmbeddingNet(bb, head, pretrained=False, metric=metric).eval()
    r = m.res
    row = {"backbone": bb, "head": head, "metric": metric,
           "params_backbone_M": count_params(m.backbone) / 1e6,
           "params_head_M": count_params(m.head) / 1e6,
           "params_metric_M": count_params(m.metric) / 1e6 if m.metric is not None else 0.0}
    x1 = torch.rand(1, 3, r, r)
    fc = FlopCounterMode(display=False)
    with fc:
        m(x1)
    row["gflops"] = fc.get_total_flops() / 1e9
    row["gmacs"] = fc.get_total_flops() / 2e9
    torch.set_num_threads(4)
    row["cpu_ms_b1"] = _latency(lambda: m(x1), n_iter=10)
    if device.type == "cuda":
        mg = m.to(device)
        xg1, xg32 = x1.to(device), torch.rand(32, 3, r, r, device=device)
        sync = torch.cuda.synchronize
        with torch.autocast("cuda", dtype=torch.float16):
            row["gpu_ms_b1"] = _latency(lambda: mg(xg1), sync=sync)
            row["gpu_ms_b32"] = _latency(lambda: mg(xg32), sync=sync)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            mg(xg32)
        row["gpu_peak_mem_mb_b32"] = torch.cuda.max_memory_allocated() / 2 ** 20
        if mg.metric is not None:
            q, c = torch.randn(75, mg.d_emb, device=device), torch.randn(5, mg.d_emb, device=device)
            row["metric_ms_75x5"] = _latency(lambda: mg.metric(q, c), sync=sync)
        m = mg.cpu()
    return row


def run_all(device):
    rows = []
    for bb, h, mt in COMBOS:
        try:
            rows.append(measure(bb, h, mt, device))
            print(rows[-1], flush=True)
        except Exception as e:  # keep going; report the failure
            rows.append({"backbone": bb, "head": h, "metric": mt, "error": repr(e)})
    return rows
