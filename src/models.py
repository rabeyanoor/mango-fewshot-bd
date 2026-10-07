"""Backbones, projection heads (identity / MLP / KAN) and the embedding model.

Embedding model:  image -> backbone features f (d_f) -> head g -> z (d_z)
The head is the only part that changes between the "MLP" and "KAN" arms of
the study, so every other factor (backbone, objective, data, schedule) is
held fixed when the two are compared.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Kolmogorov-Arnold layer (B-spline KAN, Liu et al. 2024; vectorised as in
# the "efficient-kan" formulation).  Each edge i->j carries
#     phi_ij(x) = w_b * silu(x) + w_s * sum_k c_ijk B_k(x)
# with cubic B-splines on a uniform grid.
# --------------------------------------------------------------------------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0), scale_noise=0.1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.spline_weight.uniform_(-scale_noise / grid_size, scale_noise / grid_size)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x):
        # x: (B, in) -> (B, in, grid_size + spline_order)
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[..., :-1]
                     + (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * bases[..., 1:])
        return bases

    def forward(self, x):
        x = x.float()  # spline recursion is not fp16-safe
        base = F.linear(F.silu(x), self.base_weight)
        w = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(self.b_splines(x).flatten(1), w.flatten(1))
        return base + spline


class KANHead(nn.Module):
    """LayerNorm keeps inputs inside the spline grid; two KAN layers."""

    def __init__(self, d_in, d_hidden=256, d_out=128, grid_size=5):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.l1 = KANLinear(d_in, d_hidden, grid_size=grid_size)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.l2 = KANLinear(d_hidden, d_out, grid_size=grid_size)

    def forward(self, x):
        return self.l2(self.norm2(self.l1(self.norm(x))))


class MLPHead(nn.Module):
    def __init__(self, d_in, d_hidden=256, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x.float())


class KANMetric(nn.Module):
    """Learnable additive dissimilarity between a query q and a prototype c.

        d(q, c) = ||q - c||^2 + sum_j phi_j(|q_j - c_j| * sqrt(D))

    A single KAN layer with one output is exactly a sum of learned univariate
    functions (the inner sum of the Kolmogorov-Arnold representation), so the
    metric generalises (weighted) squared Euclidean while every phi_j can be
    plotted and inspected.  The KAN term starts at ~0, i.e. training begins
    from a standard Prototypical Network.
    """

    def __init__(self, d, grid_size=5):
        super().__init__()
        self.d = d
        self.kan = KANLinear(d, 1, grid_size=grid_size, grid_range=(0.0, 4.0), scale_noise=0.01)
        with torch.no_grad():
            self.kan.base_weight.zero_()
            self.kan.spline_scaler.mul_(0.1)

    def forward(self, q, c):
        """q: (Q, D), c: (N, D) normalised embeddings -> (Q, N) dissimilarities."""
        with torch.autocast("cuda", enabled=False):
            q, c = q.float(), c.float()
            delta = q[:, None, :] - c[None, :, :]
            euc = delta.pow(2).sum(-1)
            u = (delta.abs() * self.d ** 0.5).clamp(max=3.99).flatten(0, 1)
            return euc + self.kan(u).view(q.shape[0], c.shape[0])


def make_head(kind, d_in, d_hidden=256, d_out=128):
    if kind == "none":
        return nn.Identity(), d_in
    if kind == "mlp":
        return MLPHead(d_in, d_hidden, d_out), d_out
    if kind == "kan":
        return KANHead(d_in, d_hidden, d_out), d_out
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------
class Conv4(nn.Module):
    """The original Prototypical-Network encoder (Snell et al. 2017)."""

    def __init__(self, hid=64):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(block(3, hid), block(hid, hid), block(hid, hid), block(hid, hid))

    def forward(self, x):
        return self.net(x).flatten(1)


BACKBONES = {
    # name: (input resolution, backbone lr)
    "conv4": (84, 1e-3),
    "resnet18": (224, 1e-4),
    "resnet50": (224, 1e-4),
    "densenet121": (224, 1e-4),
    "dinov2_s": (224, 2e-5),
    "dinov2_b": (224, 2e-5),
    # lightweight CNNs used by earlier mango-variety classifiers
    "efficientnet_b0": (224, 1e-4),
    "mobilenet_v2": (224, 1e-4),
}


def make_backbone(name, pretrained=True):
    import torchvision.models as tvm
    if name == "conv4":
        m = Conv4()
        return m, 64 * 5 * 5
    if name == "resnet18":
        m = tvm.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        d = m.fc.in_features
        m.fc = nn.Identity()
        return m, d
    if name == "resnet50":
        m = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        d = m.fc.in_features
        m.fc = nn.Identity()
        return m, d
    if name == "densenet121":
        m = tvm.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        d = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, d
    if name in ("efficientnet_b0", "mobilenet_v2"):
        m = getattr(tvm, name)(weights="IMAGENET1K_V1" if pretrained else None)
        d = m.classifier[-1].in_features
        m.classifier = nn.Identity()
        return m, d
    if name == "dinov2_s":
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
                              num_classes=0, img_size=224)
        return m, m.num_features
    if name == "dinov2_b":
        import timm
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=pretrained,
                              num_classes=0, img_size=224)
        m.set_grad_checkpointing(True)  # fits a P x M = 80 batch on a 16 GB T4
        return m, m.num_features
    raise ValueError(name)


class EmbeddingNet(nn.Module):
    def __init__(self, backbone="resnet18", head="none", pretrained=True,
                 d_hidden=256, d_out=128, metric="euclid"):
        super().__init__()
        self.res, self.backbone_lr = BACKBONES[backbone]
        self.backbone, self.d_feat = make_backbone(backbone, pretrained)
        self.head, self.d_emb = make_head(head, self.d_feat, d_hidden, d_out)
        self.metric = KANMetric(self.d_emb) if metric == "kan" else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def features(self, x):
        """x: float images in [0,1], already at self.res."""
        return self.backbone((x - self.mean) / self.std).float()

    def forward(self, x, return_feat=False):
        f = self.features(x)
        with torch.autocast("cuda", enabled=False):  # KAN splines need fp32
            z = self.head(f.float()).float()
        return (z, f) if return_feat else z

    def param_groups(self, head_lr):
        heads = list(self.head.parameters())
        if self.metric is not None:
            heads += list(self.metric.parameters())
        return [{"params": self.backbone.parameters(), "lr": self.backbone_lr},
                {"params": heads, "lr": head_lr}]


class FusionNet(nn.Module):
    """Feature-level fusion of several embedding networks, used for evaluation only.

    The backbone features of the members are L2-normalised and concatenated, so each
    member contributes equally; the concatenation is both z and f of the fused model.
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.res = members[0].res
        assert all(m.res == self.res for m in members)
        self.metric = None

    def forward(self, x, return_feat=False):
        f = torch.cat([F.normalize(m.features(x).float(), dim=1) for m in self.members], 1)
        z = F.normalize(f, dim=1)
        return (z, f) if return_feat else z


def count_params(m):
    return sum(p.numel() for p in m.parameters())
