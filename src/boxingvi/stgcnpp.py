"""ST-GCN++ (PYSKL), ported to plain PyTorch so its pretrained weights load
without mmcv.

Why this exists: ADR-0008 committed contribution #1 to ST-GCN++, and the plain
ST-GCN in stgcn.py already memorises 3,781 clips (0.921 train vs 0.558 val). A
larger model from scratch would overfit harder. A model pretrained on a large
action corpus and fine-tuned here is the version that tests the learning-curve
claim -- that the random forest wins because data is short, not because the
architecture is wrong.

The architecture and parameter names follow PYSKL exactly
(pyskl/models/gcns/stgcn.py, utils/gcn.py, utils/tcn.py, utils/graph.py, config
configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py), so a PYSKL state_dict maps onto
this module key for key. PYSKL itself is not a dependency: it needs mmcv, which
is a poor fit for a CPU-only Windows machine.

Trust boundary: the checkpoint is a pickle fetched from the network. It is only
ever opened with ``weights_only=True``, its size and digest are checked first,
and every backbone tensor must match this architecture by name and shape.
Anything else raises. There is no fallback to an unsafe load.
"""

from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import N_CLASSES, N_JOINTS

EPS = 1e-4

# PYSKL model zoo, HRNet 2D skeletons in COCO-17 order -- the same joint layout
# BoxingVI ships. Size and md5 come from the host's Content-Length and ETag
# (a plain, non-multipart ETag on this host is the MD5 of the object). sha256 is
# pinned after the first verified download; see ISSUES-LOG.
PRETRAINED = {
    "ntu120_xsub_hrnet_j": {
        "url": "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/"
               "stgcnpp_ntu120_xsub_hrnet/j.pth",
        "size": 5_915_801,
        "md5": "c61f2dd4e71f130ee3e942580356cf26",
        # Pinned 2026-09-16 after a verified download: size and md5 matched the
        # host, and all 690 backbone tensors matched this port by name and shape.
        "sha256": "c6e074210fa30c01081460665ce9c1c221ba4a60d7eeb10456cde6e8ac40486e",
        "source_classes": 120,
    },
    "ntu60_xsub_hrnet_j": {
        "url": "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/"
               "stgcnpp_ntu60_xsub_hrnet/j.pth",
        "size": 5_854_105,
        "md5": "9f784ef3ddef94b582aa9a619a5091e6",
        "sha256": None,
        "source_classes": 60,
    },
}

# PYSKL's COCO layout, verbatim (pyskl/utils/graph.py).
COCO_INWARD = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
]


# ---------------------------------------------------------------- graph ------

def _edge2mat(links, n: int) -> np.ndarray:
    a = np.zeros((n, n))
    for i, j in links:
        a[j, i] = 1
    return a


def _normalize_digraph(a: np.ndarray) -> np.ndarray:
    deg = a.sum(0)
    dn = np.zeros_like(a)
    for i in range(a.shape[1]):
        if deg[i] > 0:
            dn[i, i] = 1.0 / deg[i]
    return a @ dn


def coco_spatial_adjacency() -> np.ndarray:
    """(3, 17, 17): self-links, inward, outward -- PYSKL's 'spatial' mode."""
    n = N_JOINTS
    outward = [(j, i) for (i, j) in COCO_INWARD]
    return np.stack([
        _edge2mat([(i, i) for i in range(n)], n),
        _normalize_digraph(_edge2mat(COCO_INWARD, n)),
        _normalize_digraph(_edge2mat(outward, n)),
    ]).astype(np.float32)


# ---------------------------------------------------------------- units ------

class UnitGCN(nn.Module):
    """unit_gcn with adaptive='init', conv_pos='pre', with_res=True."""

    def __init__(self, cin: int, cout: int, a: torch.Tensor):
        super().__init__()
        self.num_subsets = a.size(0)
        self.A = nn.Parameter(a.clone())
        self.conv = nn.Conv2d(cin, cout * a.size(0), 1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        if cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        else:
            self.down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        res = x if self.down is None else self.down(x)
        x = self.conv(x).view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, self.A).contiguous()
        return self.act(self.bn(x) + res)


class UnitTCN(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int = 9, stride: int = 1,
                 dilation: int = 1, norm: bool = True):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(cout) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MSTCN(nn.Module):
    """Multi-scale temporal convolution: four dilated branches, a max-pool
    branch and a 1x1 branch, concatenated and mixed."""

    MS_CFG = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        nb = len(self.MS_CFG)
        mid = cout // nb
        rem = cout - mid * (nb - 1)
        act = nn.ReLU()
        branches = []
        for i, cfg in enumerate(self.MS_CFG):
            bc = rem if i == 0 else mid
            if cfg == "1x1":
                branches.append(nn.Conv2d(cin, bc, kernel_size=1, stride=(stride, 1)))
            elif cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(cin, bc, kernel_size=1), nn.BatchNorm2d(bc), act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1),
                                 padding=(1, 0))))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(cin, bc, kernel_size=1), nn.BatchNorm2d(bc), act,
                    UnitTCN(bc, bc, kernel_size=cfg[0], stride=stride,
                            dilation=cfg[1], norm=False)))
        self.branches = nn.ModuleList(branches)
        tin = mid * (nb - 1) + rem
        self.transform = nn.Sequential(nn.BatchNorm2d(tin), act,
                                       nn.Conv2d(tin, cout, kernel_size=1))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        return self.bn(self.transform(feat))


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, a: torch.Tensor, stride: int = 1,
                 residual: bool = True):
        super().__init__()
        self.gcn = UnitGCN(cin, cout, a)
        self.tcn = MSTCN(cout, cout, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self._res_mode = "zero"
        elif cin == cout and stride == 1:
            self._res_mode = "identity"
        else:
            self._res_mode = "tcn"
            self.residual = UnitTCN(cin, cout, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._res_mode == "zero":
            res = 0
        elif self._res_mode == "identity":
            res = x
        else:
            res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


class Backbone(nn.Module):
    """10 stages, base width 64, doubling and striding at stages 5 and 8."""

    def __init__(self, in_channels: int = 3, base: int = 64, num_stages: int = 10,
                 inflate_stages=(5, 8), down_stages=(5, 8)):
        super().__init__()
        a = torch.tensor(coco_spatial_adjacency())
        self.data_bn = nn.BatchNorm1d(in_channels * N_JOINTS)
        modules = [Block(in_channels, base, a.clone(), 1, residual=False)]
        inflate = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            cin = base
            if i in inflate_stages:
                inflate += 1
            cout = int(64 * 2 ** inflate + EPS)
            base = cout
            modules.append(Block(cin, cout, a.clone(), stride))
        self.gcn = nn.ModuleList(modules)
        self.out_channels = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, m, t, v, c = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()               # N M V C T
        x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        for blk in self.gcn:
            x = blk(x)
        return x.reshape((n, m) + x.shape[1:])                  # N M C T V


class Head(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc_cls = nn.Linear(in_channels, n_classes)
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.zeros_(self.fc_cls.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(3, 4)).mean(dim=1)                      # pool T, V, then persons
        return self.fc_cls(self.dropout(x))


class STGCNPP(nn.Module):
    """Attribute names mirror PYSKL's RecognizerGCN (backbone / cls_head), so
    checkpoint keys map across unchanged."""

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.0):
        super().__init__()
        self.backbone = Backbone()
        self.cls_head = Head(self.backbone.out_channels, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.shape[3] != N_JOINTS or x.shape[4] != 3:
            raise ValueError(f"expected (N, M, T, {N_JOINTS}, 3), got {tuple(x.shape)}")
        return self.cls_head(self.backbone(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------- input ------

def prepare_input(x: np.ndarray, t_out: int = 50) -> torch.Tensor:
    """(N, T, 17, 3) BoxingVI arrays -> (N, 1, t_out, 17, 3) in PYSKL's space.

    BoxingVI stores image-normalised coordinates in [0, 1] and a validity flag
    in channel 2. PYSKL was pretrained on coordinates in [-1, 1] (PreNormalize2D,
    fixed image size) with a detector score in channel 2, and never on padding:
    UniformSample draws from the real frames only. So:

    * keep only frames with at least one observed joint, and resample them
      uniformly to t_out -- no padding reaches the network;
    * map x, y to 2v - 1, and zero any joint that was not observed, as PYSKL
      does below its score threshold;
    * pass the validity flag through as the score channel.

    Raises rather than guessing on anything that would silently wreck transfer:
    wrong shape, non-finite values, a clip with no observed frame, or
    coordinates far outside [0, 1] -- pixel-space input would otherwise be
    fed straight through and the pretrained weights would see nonsense.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 4:
        raise ValueError(f"expected a 4-D array (N, T, V, C), got {getattr(x, 'shape', type(x))}")
    if x.shape[2] != N_JOINTS or x.shape[3] != 3:
        raise ValueError(f"expected {N_JOINTS} joints x 3 channels, got {x.shape[2:]}")
    if t_out < 1:
        raise ValueError("t_out must be positive")
    if not np.isfinite(x).all():
        raise ValueError("input contains NaN or inf")

    out = np.zeros((x.shape[0], 1, t_out, N_JOINTS, 3), dtype=np.float32)
    for i, clip in enumerate(x):
        mask = clip[..., 2] > 0                                   # (T, V)
        frames = np.flatnonzero(mask.any(axis=1))
        if frames.size == 0:
            raise ValueError(f"clip {i} has no observed frame")
        xy = clip[frames, :, :2][mask[frames]]
        # Pose estimators legitimately place a joint a little past the frame
        # edge, so small overshoot is normal. The guard is for scale: pixel
        # coordinates run into the hundreds.
        if xy.size and (xy.min() < -0.5 or xy.max() > 1.5):
            raise ValueError(
                f"clip {i} has coordinates far outside [0, 1] ({xy.min():.3g}..{xy.max():.3g}); "
                "BoxingVI is image-normalised, so this is probably pixel-space input")
        pick = frames[np.round(np.linspace(0, frames.size - 1, t_out)).astype(int)]
        sel = clip[pick]                                          # (t_out, V, 3)
        m = sel[..., 2] > 0
        out[i, 0, ..., 0] = np.where(m, sel[..., 0] * 2 - 1, 0.0)
        out[i, 0, ..., 1] = np.where(m, sel[..., 1] * 2 - 1, 0.0)
        out[i, 0, ..., 2] = m.astype(np.float32)
    return torch.from_numpy(out)


# ----------------------------------------------------------- checkpoint ------

# Real checkpoints are a few MB; this bounds the read when no spec pins a size.
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def _digest(blob: bytes, algo: str) -> str:
    return hashlib.new(algo, blob).hexdigest()


def load_pretrained(model: STGCNPP, path: str | Path, spec: Mapping | None = None) -> dict:
    """Load a PYSKL ST-GCN++ backbone into `model`. The classifier head is left
    as initialised, since the source task has a different label set.

    Checks, in order, and raises on the first failure: the file exists and is a
    regular file; its size and digests match `spec` where given; it opens under
    ``weights_only=True``; it holds a mapping of tensors; the backbone keys
    match this architecture exactly, with no missing or unexpected names; and
    every shape matches.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    want_size = spec.get("size") if spec else None
    # Size is checked on the stat first, so a wrong or oversized file is refused
    # before it is read into memory, and again on the bytes themselves.
    size = path.stat().st_size
    if want_size is not None and size != want_size:
        raise ValueError(f"checkpoint size {size} != expected {want_size}")
    if size > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint is {size} bytes, over the {MAX_CHECKPOINT_BYTES} limit")
    # Read once. Every digest is taken over these bytes, and these bytes are what
    # gets unpickled, so the file cannot change between the check and the load.
    blob = path.read_bytes()
    if want_size is not None and len(blob) != want_size:
        raise ValueError(f"checkpoint size {len(blob)} != expected {want_size}")
    if spec:
        for algo in ("md5", "sha256"):
            want = spec.get(algo)
            if want and _digest(blob, algo) != want:
                raise ValueError(f"checkpoint {algo} does not match the pinned value")

    try:
        ckpt = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as e:  # never fall back to an unsafe load
        raise ValueError(f"checkpoint refused by the safe loader: {type(e).__name__}") from None

    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, Mapping) else None
    if not isinstance(sd, Mapping) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise ValueError("checkpoint does not contain a state_dict of tensors")

    want = {k: v for k, v in model.state_dict().items() if k.startswith("backbone.")}
    got = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    if missing or unexpected:
        raise ValueError(f"backbone keys differ: {len(missing)} missing "
                         f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                         f"(e.g. {unexpected[:3]})")
    bad = [k for k in want if want[k].shape != got[k].shape]
    if bad:
        raise ValueError(f"shape mismatch on {len(bad)} tensors, e.g. {bad[0]}: "
                         f"{tuple(got[bad[0]].shape)} vs {tuple(want[bad[0]].shape)}")

    model.load_state_dict({**model.state_dict(), **copy.copy(got)}, strict=True)
    src_w = sd.get("cls_head.fc_cls.weight")
    return {
        "loaded_tensors": len(got),
        "source_classes": int(src_w.shape[0]) if isinstance(src_w, torch.Tensor) else None,
        "sha256": _digest(blob, "sha256"),
    }
