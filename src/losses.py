"""Training objectives compared in the study.

All objectives consume the same class-balanced batch of P classes x M images
(see data.PKSampler), so they see exactly the same images per step.

  ce        softmax cross-entropy with a cosine classifier (Baseline++)
  arcface   additive angular margin (Deng et al. 2019)
  supcon    supervised contrastive (Khosla et al. 2020)
  triplet   triplet margin loss with a selectable mining policy
  proto     Prototypical-Network episodic loss (Snell et al. 2017)
  proto_tri proto + lambda * triplet (aggregated metric objective)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist2(a, b):
    """Squared euclidean distance matrix."""
    return (a.pow(2).sum(1, keepdim=True) - 2 * a @ b.t() + b.pow(2).sum(1).unsqueeze(0)).clamp_min(0)


# ---------------------------------------------------------------- triplet
def triplet_loss(z, y, margin=0.2, mining="batch_hard", cutoff=0.5):
    """z: L2-normalised embeddings (B, d); y: (B,) labels."""
    d = pdist2(z, z).clamp_min(1e-12).sqrt()
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos_mask, neg_mask = same & ~eye, ~same
    if mining == "batch_hard":
        hp = (d * pos_mask).max(1).values
        hn = d.masked_fill(~neg_mask, float("inf")).min(1).values
        return F.relu(hp - hn + margin).mean()
    # all anchor-positive pairs
    a_idx, p_idx = pos_mask.nonzero(as_tuple=True)
    d_ap = d[a_idx, p_idx]
    d_an = d[a_idx]  # (n_pairs, B)
    nm = neg_mask[a_idx]
    if mining == "random":
        w = nm.float()
    elif mining == "semi_hard":
        # negatives farther than the positive but inside the margin;
        # fall back to the hardest negative when none qualify
        semi = nm & (d_an > d_ap[:, None]) & (d_an < d_ap[:, None] + margin)
        none = ~semi.any(1)
        hard = d_an.masked_fill(~nm, float("inf")).argmin(1)
        semi[none, hard[none]] = True
        w = semi.float()
    elif mining == "distance_weighted":
        # Wu et al. 2017: sample negatives with prob ~ 1/q(d) on the unit sphere
        dim = z.shape[1]
        dc = d_an.clamp_min(cutoff)
        log_w = (2.0 - dim) * dc.log() - ((dim - 3) / 2.0) * (1.0 - 0.25 * dc.pow(2)).clamp_min(1e-8).log()
        log_w = log_w.masked_fill(~nm, float("-inf"))
        w = torch.softmax(log_w - log_w.max(1, keepdim=True).values, 1)
        w = w * (d_an < 1.4).float() + 1e-12 * nm.float()
    else:
        raise ValueError(mining)
    neg = torch.multinomial(w / w.sum(1, keepdim=True), 1).squeeze(1)
    return F.relu(d_ap - d_an.gather(1, neg[:, None]).squeeze(1) + margin).mean()


# ------------------------------------------------------------------ proto
def proto_logits(zs, ys, zq, n_way, scale, metric=None):
    protos = torch.stack([zs[ys == c].mean(0) for c in range(n_way)])
    d = pdist2(zq, protos) if metric is None else metric(zq, protos)
    return -scale * d


def proto_loss(z, y, n_support, scale, metric=None, margin=0.0):
    """Split every class of the P x M batch into support / query halves.

    margin > 0 gives the prototype margin loss: the true-class distance is
    inflated by `margin` inside the softmax, so a query must be closer to its
    own prototype than to every other prototype by at least that much."""
    if z.device.type == "xla":
        return _proto_loss_static(z, y, n_support, scale, metric, margin)
    classes = y.unique()
    ys, yq, zs, zq = [], [], [], []
    for i, c in enumerate(classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        zs.append(z[idx[:n_support]]); ys += [i] * n_support
        zq.append(z[idx[n_support:]]); yq += [i] * (len(idx) - n_support)
    zs, zq = torch.cat(zs), torch.cat(zq)
    ys = torch.tensor(ys, device=z.device)
    yq = torch.tensor(yq, device=z.device)
    logits = proto_logits(zs, ys, zq, len(classes), scale, metric)
    acc = (logits.argmax(1) == yq).float().mean()
    if margin > 0:
        logits = logits - scale * margin * F.one_hot(yq, len(classes))
    return F.cross_entropy(logits, yq), acc


def _proto_loss_static(z, y, n_support, scale, metric, margin):
    """proto_loss with shapes fixed by the P x M batch (no data-dependent indexing),
    for XLA devices. The support / query split is computed on the host."""
    yc = y.cpu()
    classes = yc.unique()
    sup, qry, yq = [], [], []
    for i, c in enumerate(classes):
        idx = (yc == c).nonzero(as_tuple=True)[0]
        sup.append(idx[:n_support]); qry.append(idx[n_support:]); yq += [i] * (len(idx) - n_support)
    n_way = len(classes)
    zs = z[torch.cat(sup).to(z.device)]
    zq = z[torch.cat(qry).to(z.device)]
    yq = torch.tensor(yq, device=z.device)
    protos = zs.view(n_way, n_support, -1).mean(1)
    logits = -scale * (pdist2(zq, protos) if metric is None else metric(zq, protos))
    acc = (logits.argmax(1) == yq).float().mean()
    if margin > 0:
        logits = logits - scale * margin * F.one_hot(yq, n_way)
    return F.cross_entropy(logits, yq), acc


# ----------------------------------------------------------------- supcon
def supcon_loss(z, y, t=0.1):
    sim = z @ z.t() / t
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    return -(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp_min(1)).mean()


# ------------------------------------------------------- classifier heads
class CosineClassifier(nn.Module):
    def __init__(self, d, n_classes, scale=16.0, margin=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, d) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, z, y=None):
        cos = F.normalize(z, dim=1) @ F.normalize(self.w, dim=1).t()
        if self.margin > 0 and y is not None:  # ArcFace: cos(theta + m) on the target
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            target = torch.cos(theta + self.margin)
            onehot = F.one_hot(y, cos.shape[1]).bool()
            cos = torch.where(onehot, target, cos)
        return self.scale * cos


class Objective(nn.Module):
    def __init__(self, kind, d_emb, n_classes, n_support=5, tri_margin=0.2,
                 mining="batch_hard", lam=1.0, margin=0.0):
        super().__init__()
        self.kind, self.n_support, self.lam, self.margin = kind, n_support, lam, margin
        self.tri_margin, self.mining = tri_margin, mining
        self.log_scale = nn.Parameter(torch.tensor(2.3))  # learnable temperature, exp(2.3)~10
        if kind in ("ce", "arcface"):
            self.clf = CosineClassifier(d_emb, n_classes, margin=0.3 if kind == "arcface" else 0.0)

    def forward(self, z, y, metric=None):
        """z: raw embeddings, y: global class ids, metric: the model's learned
        distance (None = squared Euclidean). returns (loss, train-acc)."""
        zn = F.normalize(z, dim=1)
        k = self.kind
        if k in ("ce", "arcface"):
            logits = self.clf(z, y)
            acc = (self.clf(z).argmax(1) == y).float().mean()
            return F.cross_entropy(logits, y), acc
        if k == "supcon":
            return supcon_loss(zn, y), torch.tensor(0.0)
        if k == "triplet":
            return triplet_loss(zn, y, self.tri_margin, self.mining), torch.tensor(0.0)
        if k in ("proto", "proto_tri"):
            loss, acc = proto_loss(zn, y, self.n_support, self.log_scale.exp(), metric, self.margin)
            if k == "proto_tri":
                loss = loss + self.lam * triplet_loss(zn, y, self.tri_margin, self.mining)
            return loss, acc
        raise ValueError(k)
