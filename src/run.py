"""Experiment driver used inside Kaggle kernels.

    import run; run.main("suite_name")

* finds the image cache (output of the prep kernel) under /kaggle/input
* expands the suite into a list of configs
* shards configs over all visible GPUs (one worker process per GPU)
* appends one JSON line per finished config to /kaggle/working/results_<gpu>.jsonl
* skips configs whose key already appears in any results*.jsonl under
  /kaggle/input, so a suite can be resumed by attaching the previous run.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

# Kaggle paths by default; override to run the same driver locally.
INPUT = os.environ.get("MANGOFS_INPUT", "/kaggle/input")
WORK = os.environ.get("MANGOFS_WORK", "/kaggle/working")


def find_root(variant="crop"):
    """Folder of the image cache that holds <variant>.npy next to meta.csv."""
    hits = [os.path.dirname(p) for p in glob.glob(f"{INPUT}/**/meta.csv", recursive=True)]
    hits = [h for h in hits if os.path.exists(os.path.join(h, f"{variant}.npy"))] or hits
    if not hits:
        sys.exit(f"image cache (meta.csv) not found in {INPUT}")
    return hits[0]


def find_groups():
    hits = glob.glob(f"{INPUT}/**/groups.csv", recursive=True) + glob.glob(f"{WORK}/groups.csv")
    return hits[0] if hits else None


def done_keys():
    keys = set()
    for p in glob.glob(f"{INPUT}/**/results*.jsonl", recursive=True) + glob.glob(f"{WORK}/results*.jsonl"):
        with open(p) as f:
            for line in f:
                try:
                    keys.add(json.loads(line)["key"])
                except Exception:
                    pass
    return keys


# ---------------------------------------------------------------- suites
BACKBONES = ["conv4", "resnet18", "resnet50", "densenet121", "dinov2_s"]

# Named methods.  Everything that is not the proposed method is a baseline
# or an ablation that isolates one ingredient of it.
METHODS = {
    "ce":            dict(objective="ce", head="mlp"),               # fine-tuned classifier (Baseline++)
    "proto":         dict(objective="proto", head="none"),           # vanilla ProtoNet
    "proto_mlp":     dict(objective="proto", head="mlp"),            # ProtoNet + MLP projection
    "proto_kanhead": dict(objective="proto", head="kan"),            # ProtoNet + KAN projection
    "proto_margin":  dict(objective="proto", head="mlp", margin=0.1),  # + margin only
    "kan_metric":    dict(objective="proto", head="mlp", metric="kan"),  # + KAN metric only
    "ours":          dict(objective="proto", head="mlp", metric="kan", margin=0.1),
    # extra baselines: other metric-learning objectives
    "arcface":       dict(objective="arcface", head="mlp"),
    "supcon":        dict(objective="supcon", head="mlp"),
    "triplet":       dict(objective="triplet", head="mlp"),
    "proto_tri":     dict(objective="proto_tri", head="mlp"),
    "ours_kanhead":  dict(objective="proto", head="kan", metric="kan", margin=0.1),
}
MAIN = ["ce", "proto", "proto_mlp", "proto_kanhead", "proto_margin", "kan_metric", "ours"]
# backbones whose features are concatenated in the fusion evaluation (all pretrained ones)
FUSION = ["resnet18", "resnet50", "densenet121", "dinov2_s", "dinov2_b"]
LIGHT = ["efficientnet_b0", "mobilenet_v2"]
EXTRA = ["arcface", "supcon", "triplet", "proto_tri", "ours_kanhead"]


def steps_for(bb):
    return 3000 if bb == "conv4" else 1200


def m(method, **kw):
    return dict(METHODS[method], method=method, **kw)


def suite(name):
    U = dict(protocol="unified")
    if name == "groups":
        return []  # main() builds groups.csv before dispatching
    if name == "smoke":
        return [m(x, **U, fold=0, backbone="resnet18", steps=60, n_episodes=50)
                for x in ["proto_mlp", "ours", "proto_kanhead"]] + \
               [dict(kind="zeroshot", **U, fold=0, backbone="resnet18", n_episodes=50)]
    if name == "unified_main":
        cfgs = []
        for fold in range(3):
            for bb in BACKBONES:
                if bb != "conv4":
                    cfgs.append(dict(kind="zeroshot", **U, fold=fold, backbone=bb))
                for x in MAIN:
                    cfgs.append(m(x, **U, fold=fold, backbone=bb, steps=steps_for(bb)))
        return cfgs
    if name == "unified_extra":
        cfgs = [m(x, **U, fold=f, backbone=bb) for f in range(3)
                for bb in ["resnet18", "densenet121"] for x in EXTRA]
        # margin sensitivity
        cfgs += [dict(m("ours", **U, fold=f, backbone="resnet18"), margin=mg)
                 for f in range(3) for mg in (0.05, 0.2, 0.4)]
        # triplet mining policies (RQ3)
        cfgs += [dict(m("triplet", **U, fold=f, backbone="resnet18"), mining=mn)
                 for f in range(3) for mn in ("random", "semi_hard", "distance_weighted")]
        return cfgs
    if name == "ablations":
        cfgs = []
        for f in range(3):
            for bb in ["resnet18", "densenet121"]:
                for x in ["proto_mlp", "ours"]:
                    cfgs.append(m(x, **U, fold=f, backbone=bb, variant="full"))  # background kept
                for x in ["proto", "proto_mlp", "ours"]:
                    for s_ in (1, 2):
                        cfgs.append(m(x, **U, fold=f, backbone=bb, seed=s_))   # seed variance
        return cfgs
    # follow-up experiments after comparing with other mango-variety pipelines:
    # background removed (seg.npy, prep_seg.py), lightweight CNNs, feature fusion
    if name == "seg_zero":  # no training; runs on CPU
        cfgs = [dict(kind="zeroshot", **U, fold=f, backbone=bb, variant="seg", eval=2)
                for f in range(3) for bb in FUSION + LIGHT]
        cfgs += [dict(kind="zeroshot", **U, fold=f, backbone=bb, eval=2)
                 for f in range(3) for bb in LIGHT]
        return cfgs
    if name == "fusion":  # no training; runs on CPU
        return [dict(kind="fusion", **U, fold=f, members=mb)
                for mb in ("imagenet", "ce") for f in range(3)]
    if name == "seg_train":  # GPU
        return [m(x, **U, fold=f, backbone=bb, variant="seg", eval=2) for f in range(3)
                for bb in ["resnet18", "densenet121"] for x in ["ce", "proto", "proto_mlp"]]
    if name == "light_train":  # GPU
        return [m(x, **U, fold=f, backbone=bb, eval=2) for f in range(3)
                for bb in LIGHT for x in ["ce", "proto", "proto_mlp"]]
    if name == "lodo_main":
        cfgs = []
        for fold in range(3):
            for bb in ["resnet18", "densenet121", "resnet50", "dinov2_s"]:
                cfgs.append(dict(kind="zeroshot", protocol="lodo", fold=fold, backbone=bb))
                for x in ["ce", "proto_mlp", "ours"]:
                    cfgs.append(m(x, protocol="lodo", fold=fold, backbone=bb))
        return cfgs
    if name == "improve":
        # (1) re-evaluate the saved models with the backbone-feature and source-centred
        #     rules (identical episodes; no retraining)
        cfgs = [dict(kind="reeval", ckpt=ck) for ck in input_ckpts(
            lambda c: c["backbone"] in BACKBONES and c.get("variant", "crop") == "crop"
            and c.get("seed", 0) == 0 and c["method"] in ("ce", "proto", "proto_mlp", "ours"))]
        # (2) untrained features with the new rules, plus the larger DINOv2 ViT-B/14
        for p_ in ("unified", "lodo"):
            for fold in range(3):
                for bb in ["resnet18", "resnet50", "densenet121", "dinov2_s", "dinov2_b"]:
                    cfgs.append(dict(kind="zeroshot", protocol=p_, fold=fold, backbone=bb, eval=2))
        for fold in range(3):
            for x in ["proto", "proto_mlp", "ce"]:
                cfgs.append(m(x, **U, fold=fold, backbone="dinov2_b", eval=2))
        return cfgs
    if name == "episode_ft":
        return [dict(kind="episode_ft", protocol="unified", fold=f, backbone=bb)
                for f in range(3) for bb in ["resnet18", "densenet121"]]
    if name == "efficiency":
        return [dict(kind="efficiency")]
    if name == "closed_set":
        return [dict(kind="closed", dataset=ds, backbone=bb, naive=nv)
                for ds in ["MangoImageBD", "MangoClassify12", "Mangifera2012"]
                for bb in ["resnet18", "densenet121"] for nv in (False, True)]
    raise ValueError(name)


def input_records():
    for p in glob.glob(f"{INPUT}/**/results*.jsonl", recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


def input_ckpts(keep):
    """Checkpoint names of attached trained models whose config satisfies keep(cfg)."""
    out = []
    for r in input_records():
        if r.get("kind") == "train" and r.get("ckpt") and keep({**r["cfg"], "method": r["method"]}):
            out.append(r["ckpt"])
    return sorted(set(out))


def reevaluate(ca, ckpt, dev):
    import torch
    import train as T
    from data import protocol_split
    from models import EmbeddingNet
    rec = next(r for r in input_records() if r.get("ckpt") == ckpt)
    cfg = {**rec["cfg"], "eval": 2}
    path = glob.glob(f"{INPUT}/**/ckpt/{ckpt}", recursive=True)[0]
    model = EmbeddingNet(cfg["backbone"], cfg["head"], pretrained=False, metric=cfg["metric"])
    model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    model.to(dev)
    tr, te, info = protocol_split(ca.meta, cfg["protocol"], cfg["fold"])
    res, _ = T.evaluate_model(model, ca, cfg, dev, te, info, tr)
    return {"cfg": cfg, "split": info, "n_train": len(tr), "n_test": len(te),
            "params_backbone": rec["params_backbone"], "params_head": rec["params_head"],
            "train": {}, "results": res, "reeval_of": rec["key"], "method": rec["method"]}


def load_trained(protocol, fold, backbone, method):
    """Saved seed-0 crop model of the attached runs (fp16 state dict -> fp32)."""
    import torch
    from models import EmbeddingNet
    rec = next(r for r in input_records() if r.get("kind") == "train" and r.get("ckpt")
               and r["method"] == method and r["cfg"]["protocol"] == protocol
               and r["cfg"]["fold"] == fold and r["cfg"]["backbone"] == backbone
               and r["cfg"].get("seed", 0) == 0 and r["cfg"].get("variant", "crop") == "crop")
    cfg = rec["cfg"]
    path = glob.glob(f"{INPUT}/**/ckpt/{rec['ckpt']}", recursive=True)[0]
    model = EmbeddingNet(cfg["backbone"], cfg["head"], pretrained=False, metric=cfg["metric"])
    model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return model, rec["ckpt"]


def fusion_eval(ca, c, dev):
    import train as T
    from data import protocol_split
    from models import EmbeddingNet, FusionNet, count_params
    if c["members"] == "imagenet":
        members, src = [EmbeddingNet(b, "none", True) for b in FUSION], []
    else:
        pairs = [load_trained(c["protocol"], c["fold"], b, c["members"]) for b in FUSION]
        members, src = [p[0] for p in pairs], [p[1] for p in pairs]
    model = FusionNet(members).to(dev)
    cfg = {**T.DEFAULTS, **c, "backbone": "fusion", "head": "none", "eval": 2}
    tr, te, info = protocol_split(ca.meta, cfg["protocol"], cfg["fold"])
    res, _ = T.evaluate_model(model, ca, cfg, dev, te, info, tr)
    return {"cfg": cfg, "split": info, "n_train": 0, "n_test": len(te),
            "params_backbone": count_params(model), "params_head": 0, "train": {},
            "results": res, "members": FUSION, "ckpts": src,
            "method": f"fusion_{c['members']}"}


def on_tpu():
    return bool(os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("PJRT_DEVICE") == "TPU")


def pick_device():
    """CUDA if present, else the TPU (torch_xla) of a TPU VM, else CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if on_tpu():
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device("cpu")


def ckey(c):
    return json.dumps(c, sort_keys=True)


# ---------------------------------------------------------------- worker
def worker(name, shard, n_shards):
    import torch
    from data import Cache
    import train as T
    dev = pick_device()
    cache_by_variant = {}

    def cache(variant="crop"):
        if variant not in cache_by_variant:
            cache_by_variant[variant] = Cache(find_root(variant), variant, find_groups())
        return cache_by_variant[variant]
    done = done_keys()
    cfgs = suite(name)
    mine = [c for i, c in enumerate(cfgs) if i % n_shards == shard]
    out = open(f"{WORK}/results_{name}_{shard}.jsonl", "a")
    os.makedirs(f"{WORK}/ckpt", exist_ok=True)
    for i, c in enumerate(mine):
        c = dict(c)
        kind = c.pop("kind", "train")
        method = c.pop("method", None)
        key = ckey({"kind": kind, "method": method, **c})
        if key in done:
            continue
        print(f"[gpu{shard}] {i + 1}/{len(mine)} {key}", flush=True)
        t0 = time.time()
        try:
            ca = cache(c.get("variant", "crop"))
            if kind == "train":
                # every trained model is kept (fp16) for re-evaluation and analysis
                tag = hashlib.md5(key.encode()).hexdigest()[:8]
                ck = f"{WORK}/ckpt/{c['protocol']}{c['fold']}_{c['backbone']}_{method}_{tag}.pt"
                r = T.run_config(ca, c, dev, save_ckpt=ck)
                r["ckpt"] = os.path.basename(ck)
            elif kind == "reeval":
                r = reevaluate(ca, c["ckpt"], dev)
            elif kind == "fusion":
                r = fusion_eval(ca, c, dev)
            elif kind == "zeroshot":
                r = T.run_zero_shot(ca, c, dev)
            elif kind == "episode_ft":
                from finetune import episode_finetune
                from data import protocol_split
                from models import EmbeddingNet
                tr, te, info = protocol_split(ca.meta, c["protocol"], c["fold"])
                r = {"results": episode_finetune(ca, te, EmbeddingNet(c["backbone"], "none", True), dev)}
            elif kind == "closed":
                from finetune import closed_set_study
                from data import DATASETS
                held = DATASETS.index(c["dataset"])
                metas = {x: p for x in ["proto_mlp", "ours"] for p in glob.glob(
                    f"{INPUT}/**/ckpt/lodo{held}_{c['backbone']}_{x}_*.pt", recursive=True)}
                r = {"rows": closed_set_study(ca, c["dataset"], dev, c["backbone"], ["mlp", "kan"],
                                              ["all"] if c["naive"] else [1, 5, 10, 20, "all"],
                                              [0, 1, 2], meta_ckpt=None if c["naive"] else metas,
                                              naive=c["naive"])}
            elif kind == "efficiency":
                from efficiency import run_all
                r = {"rows": run_all(dev)}
            r["key"] = key
            r["kind"] = kind
            r["method"] = method or r.get("method")
            r["cfg_in"] = c
            r["elapsed_s"] = time.time() - t0
            out.write(json.dumps(r) + "\n")
            out.flush()
        except Exception:
            traceback.print_exc()
            out_err = open(f"{WORK}/errors_{shard}.txt", "a")
            out_err.write(key + "\n" + traceback.format_exc() + "\n")
            out_err.close()
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def main(name, budget_h=11.0):
    import torch
    n = max(1, torch.cuda.device_count())
    if not torch.cuda.is_available() and on_tpu():
        n = int(os.environ.get("MANGOFS_TPU_WORKERS", "8"))  # one worker per chip (v5e-8)
    print("devices:", n, "suite:", name, "configs:", len(suite(name)), flush=True)
    if find_groups() is None:
        build_groups()
    procs = []
    for g in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        if not torch.cuda.is_available() and on_tpu():  # independent single-chip processes
            env.update(PJRT_DEVICE="TPU", TPU_VISIBLE_CHIPS=str(g), TPU_PROCESS_BOUNDS="1,1,1",
                       TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_PORT=str(8476 + g),
                       CLOUD_TPU_TASK_ID="0", TPU_PROCESS_ADDRESSES=f"localhost:{8476 + g}",
                       TPU_RUNTIME_METRICS_PORTS=str(8431 + g))
        code = f"import sys; sys.path.insert(0, '{os.path.dirname(__file__)}'); import run; run.worker('{name}', {g}, {n})"
        procs.append(subprocess.Popen([sys.executable, "-c", code], env=env))
    deadline = time.time() + budget_h * 3600
    t_start = time.time()
    while any(p.poll() is None for p in procs):
        if time.time() > deadline:
            print("time budget reached; stopping workers (resume by re-running)", flush=True)
            for p in procs:
                p.terminate()
            break
        time.sleep(30)
    codes = [p.returncode for p in procs]
    print("finished", codes)
    if n > 1 and all(codes) and time.time() - t_start < 600 and not torch.cuda.is_available():
        # one-process-per-chip TPU setup failed at start-up; use one process on chip 0
        print("all workers failed at start-up; retrying with one worker", flush=True)
        code = f"import sys; sys.path.insert(0, '{os.path.dirname(__file__)}'); import run; run.worker('{name}', 0, 1)"
        subprocess.run([sys.executable, "-c", code])


def build_groups():
    """Compute capture groups once (DINOv2 features + pHash + EXIF bursts)."""
    import numpy as np
    import pandas as pd
    import torch
    from data import Cache, build_groups as bg
    from evaluate import embed
    from models import EmbeddingNet
    ca = Cache(find_root(), "crop", groups_path="/nonexistent")
    dev = pick_device()
    m = EmbeddingNet("dinov2_s", "none", True).to(dev)
    _, feats = embed(m, ca, ca.meta.idx.to_numpy(), dev)
    np.save(f"{WORK}/dinov2_feats.npy", feats.numpy().astype(np.float16))
    group, drop, stats = bg(ca.meta, feats)
    g = pd.DataFrame({"idx": ca.meta.idx, "group": group, "drop": drop})
    g.to_csv(f"{WORK}/groups.csv", index=False)
    sizes = g.groupby("group").size()
    stats.update({"n_groups": int(len(sizes)), "max_group": int(sizes.max()),
                  "mean_group": float(sizes.mean()), "n_drop": int(drop.sum())})
    json.dump(stats, open(f"{WORK}/groups_stats.json", "w"), indent=1)
    print("groups:", stats, flush=True)
