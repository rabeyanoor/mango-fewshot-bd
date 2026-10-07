"""Train one embedding model on base cultivars and evaluate it on test episodes."""
import copy
import math
import time
import zlib

import numpy as np
import torch

from data import PKSampler, gpu_augment, protocol_split
from evaluate import embed, open_set_eval, run_episodes
from losses import Objective
from models import EmbeddingNet, count_params

DEFAULTS = dict(
    protocol="unified", fold=0, backbone="resnet18", head="none", objective="proto",
    steps=1200, P=8, M=10, n_support=5, head_lr=1e-3, wd=1e-4, warmup=0.05,
    mining="batch_hard", tri_margin=0.2, lam=1.0, pretrained=True, seed=0,
    variant="crop", n_episodes=600, shots=(1, 5, 10), n_way=5,
    metric="euclid", margin=0.0,
)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_model(cache, cfg, device, train_idx, log_every=200):
    set_seed(cfg["seed"])
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(cfg["seed"])
    meta = cache.meta.set_index("idx").loc[train_idx]
    names = sorted(meta.cultivar.unique())
    y_all = meta.cultivar.map({n: i for i, n in enumerate(names)}).to_numpy()
    model = EmbeddingNet(cfg["backbone"], cfg["head"], cfg["pretrained"],
                         metric=cfg["metric"]).to(device)
    obj = Objective(cfg["objective"], model.d_emb, len(names), cfg["n_support"],
                    cfg["tri_margin"], cfg["mining"], cfg["lam"], cfg["margin"]).to(device)
    groups = model.param_groups(cfg["head_lr"])
    if not cfg["pretrained"]:
        groups[0]["lr"] = cfg["head_lr"]
    groups.append({"params": obj.parameters(), "lr": cfg["head_lr"]})
    wd = 0.05 if cfg["backbone"].startswith("dino") else cfg["wd"]
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    sampler = PKSampler(y_all, cfg["P"], cfg["M"], seed=cfg["seed"])
    steps, warm = cfg["steps"], max(1, int(cfg["warmup"] * cfg["steps"]))
    hist, t0 = [], time.time()
    model.train()
    for step in range(steps):
        lr_f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_f
        pos = sampler.sample()
        x = gpu_augment(cache.batch(train_idx[pos], device), model.res, train=True)
        y = torch.as_tensor(y_all[pos], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x)
        loss, acc = obj(z.float(), y, model.metric)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        if device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        if step % log_every == 0 or step == steps - 1:
            hist.append((step, float(loss), float(acc), time.time() - t0))
            print(f"  step {step} loss {float(loss):.4f} acc {float(acc):.3f} {time.time() - t0:.0f}s", flush=True)
    return model, {"history": hist, "train_time_s": time.time() - t0, "n_base": len(names)}


def base_feature_mean(model, cache, train_idx, device, n=2000, seed=0):
    """Mean of L2-normalised backbone features over (a sample of) base images."""
    sub = np.random.default_rng(seed).choice(train_idx, min(n, len(train_idx)), replace=False)
    _, Fb = embed(model, cache, np.sort(sub), device)
    return torch.nn.functional.normalize(Fb, dim=1).mean(0)


def evaluate_model(model, cache, cfg, device, test_idx, info, train_idx, extra_eval=True):
    meta = cache.meta.set_index("idx").loc[test_idx]
    Z, Fe = embed(model, cache, test_idx, device)
    base_mean = base_feature_mean(model, cache, train_idx, device)
    names = sorted(meta.cultivar.unique())
    labels = meta.cultivar.map({n: i for i, n in enumerate(names)}).to_numpy()
    groups = meta.group.to_numpy()
    dom = meta.dataset.to_numpy()
    # transductive domain normalisation: centre each image on the mean feature of
    # its source dataset over the (unlabelled) test images of that source
    Fn = torch.nn.functional.normalize(Fe, dim=1)
    Fs = Fn.clone()
    for s_ in np.unique(dom):
        Fs[dom == s_] -= Fn[dom == s_].mean(0)
    seed_base = zlib.crc32(f'{cfg["protocol"]}|{cfg["fold"]}'.encode()) % 9973
    metric = copy.deepcopy(model.metric).cpu().eval() if model.metric is not None else None
    res = {"all": run_episodes(Z, Fe, labels, groups, names, n_way=cfg["n_way"], shots=cfg["shots"],
                               n_episodes=cfg["n_episodes"], seed=seed_base, metric=metric,
                               rules=("proto_z", "proto_f", "proto_fc", "proto_fs", "logreg_z"),
                               base_mean=base_mean, src_centred=Fs)}
    if extra_eval:
        res["open_set"] = open_set_eval(Z, labels, groups, n_way=cfg["n_way"], shots=(1, 5),
                                        n_episodes=300, seed=seed_base + 3, metric=metric)
        for nm, E in (("open_f", Fn), ("open_fs", Fs)):
            res["open_set"].update(open_set_eval(E, labels, groups, n_way=cfg["n_way"], shots=(1, 5),
                                                 n_episodes=300, seed=seed_base + 3, name=nm))
    if cfg["protocol"] == "lodo" and extra_eval:
        for part in ("seen", "novel"):
            keep = np.isin(meta.cultivar.to_numpy(), info[part])
            if len(np.unique(labels[keep])) >= 2:
                res[part] = run_episodes(Z[keep], Fe[keep], labels[keep], groups[keep], names,
                                         n_way=cfg["n_way"], shots=cfg["shots"],
                                         n_episodes=cfg["n_episodes"], seed=seed_base + 1,
                                         rules=("proto_z", "proto_f", "proto_fc", "proto_fs"), metric=metric,
                                         base_mean=base_mean, src_centred=Fs[keep])
    if cfg["protocol"] == "unified" and extra_eval:
        # cross-domain episodes: support from dataset a, query from dataset b
        for a in np.unique(dom):
            for b in np.unique(dom):
                if a == b:
                    continue
                r = run_episodes(Z, Fe, labels, groups, names, n_way=cfg["n_way"], shots=(1, 5),
                                 n_episodes=300, seed=seed_base + 7, rules=("proto_z", "proto_f", "proto_fs"),
                                 domains=dom, domain_pair=(a, b), metric=metric, src_centred=Fs)
                if r:
                    res[f"x:{a}->{b}"] = r
    return res, (Z, Fe, labels, groups, names)


def run_config(cache, cfg, device, save_ckpt=None):
    cfg = {**DEFAULTS, **cfg}
    tr, te, info = protocol_split(cache.meta, cfg["protocol"], cfg["fold"])
    t0 = time.time()
    model, tinfo = train_model(cache, cfg, device, tr)
    res, _ = evaluate_model(model, cache, cfg, device, te, info, tr)
    if save_ckpt:
        torch.save({k: (v.half() if v.is_floating_point() else v).cpu()
                    for k, v in model.state_dict().items()},
                   save_ckpt)
    return {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "split": info, "n_train": len(tr), "n_test": len(te),
        "params_backbone": count_params(model.backbone), "params_head": count_params(model.head),
        "train": tinfo, "results": res, "wall_s": time.time() - t0,
    }


def run_zero_shot(cache, cfg, device):
    """ImageNet-pretrained backbone, no training on mango data."""
    cfg = {**DEFAULTS, **cfg, "head": "none", "objective": "imagenet", "steps": 0}
    tr, te, info = protocol_split(cache.meta, cfg["protocol"], cfg["fold"])
    model = EmbeddingNet(cfg["backbone"], "none", True).to(device)
    res, _ = evaluate_model(model, cache, cfg, device, te, info, tr)
    return {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "split": info, "n_train": 0, "n_test": len(te),
            "params_backbone": count_params(model.backbone), "params_head": 0,
            "train": {}, "results": res, "wall_s": 0}
