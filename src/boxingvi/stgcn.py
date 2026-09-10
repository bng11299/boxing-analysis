"""A compact spatial-temporal graph convolutional network.

Written from scratch rather than pulled from pyskl/mmcv deliberately. Those
stacks need compilation, are pinned to older Python, and do not install on
Windows with Python 3.13 -- which would make the first published baseline on
this benchmark unreproducible for the next person who tries. Roughly 200 lines
of plain PyTorch is a better artifact for a result whose whole value is that
someone else can check it.

Architecture follows Yan et al. (AAAI 2018): alternating graph convolution over
the joint adjacency and temporal convolution along time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EDGES, N_CLASSES, N_JOINTS


def normalised_adjacency(num_nodes: int = N_JOINTS, edges=EDGES) -> torch.Tensor:
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    a = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    deg = a.sum(1)
    dinv = np.diag((deg ** -0.5).astype(np.float32))
    return torch.from_numpy(dinv @ a @ dinv)


class STGCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gconv = nn.Conv2d(cin, cout, kernel_size=1)
        self.tconv = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=(9, 1), padding=(4, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        if cin == cout and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(cout),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # x: (N, C, T, V). `mask` is (N, 1, T, 1) at this block's INPUT
        # resolution: 1 where a real frame sits, 0 over padding.
        res = self.residual(x)
        h = self.gconv(x)
        h = torch.einsum("nctv,vw->nctw", h, adj)
        if mask is not None:
            # Zero padding again here. The graph convolution and its batch-norm
            # give padded positions a non-zero value even though the input was
            # zero, and the kernel-9 temporal convolution below would then mix
            # that invention into the real frames next to it.
            h = h * mask
        h = self.tconv(h)
        return self.relu(h + res)


class STGCN(nn.Module):
    """Small by design: the corpus is ~5.5k clips of 25 frames."""

    def __init__(self, in_channels: int = 3, n_classes: int = N_CLASSES,
                 widths=(32, 64, 128), dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adj", normalised_adjacency())
        self.data_bn = nn.BatchNorm1d(in_channels * N_JOINTS)
        blocks, cin = [], in_channels
        for i, w in enumerate(widths):
            blocks.append(STGCNBlock(cin, w, stride=1 if i == 0 else 2,
                                     dropout=dropout))
            cin = w
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(cin, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as (N, T, V, C); ST-GCN wants (N, C, T, V).
        n, t, v, c = x.shape
        # Channel 2 is the validity mask. Capture it before the permutes, so
        # the pool below can ignore padding rather than average it in.
        valid = (x[..., 2].sum(-1) > 0).float()          # (N, T)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.reshape(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        for b in self.blocks:
            m_in = F.adaptive_max_pool1d(valid.unsqueeze(1), x.shape[2])
            x = b(x, self.adj, m_in.unsqueeze(-1))
            # And again on the way out, because the residual branch carries its
            # own invented values at the padded positions.
            m_out = F.adaptive_max_pool1d(valid.unsqueeze(1), x.shape[2])
            x = x * m_out.unsqueeze(-1)
        return self.head(self._masked_pool(x, valid))

    @staticmethod
    def _masked_pool(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Global average over the OBSERVED span only. x is (N, C, T', V).

        BoxingVI pads every clip to 25 frames and real punches run 7-13, so a
        plain mean(dim=(2,3)) averages the learned representation over 46-70%
        padding and divides by the padded length regardless. Shorter clips are
        diluted hardest, which penalises exactly the classes whose punches are
        quickest. The temporal convolutions downsample T, so the mask is pooled
        to match: adaptive max-pool asks "did this output position overlap any
        real input frame", which is the right question and survives a change of
        architecture.
        """
        t_out = x.shape[2]
        m = F.adaptive_max_pool1d(valid.unsqueeze(1), t_out)      # (N, 1, T')
        num = (x * m.unsqueeze(-1)).sum(dim=(2, 3))               # (N, C)
        den = (m.sum(dim=(1, 2)) * x.shape[3]).clamp(min=1.0)     # (N,)
        return num / den.unsqueeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StaticPoseBaseline(nn.Module):
    """Sees one frame per clip -- the middle OBSERVED one -- and nothing else.

    This is the baseline that makes the others mean something. A temporal model
    that fails to beat it has not shown that it uses time at all, only posture.

    "Middle observed" rather than `T // 2`, corrected 2026-09-02. BoxingVI pads
    every clip to 25 frames and real punches run 7-13, so 46-70% of each array
    is trailing zeros -- frame 12 lands in the padding for most clips, and the
    floor was reading a blank pose. It scored 0.256 accuracy against a 0.205
    majority-class floor and its loss moved 0.009 across 30 epochs: not a low
    baseline but a broken one. A floor that is too low silently flatters every
    model measured against it, which is the one thing this class exists to
    prevent. Channel 2 is the validity mask, so the observed span is known.
    """

    def __init__(self, n_classes: int = N_CLASSES, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(N_JOINTS * 3, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(hidden, n_classes),
        )

    @staticmethod
    def middle_observed_frame(x: torch.Tensor) -> torch.Tensor:
        """Index of the central non-padding frame of each clip. (N, T, V, C)."""
        valid = x[..., 2].sum(-1) > 0                      # (N, T)
        n, t = valid.shape
        counts = valid.sum(1)                              # frames observed
        # The ceil(count / 2)-th valid frame: the middle of what was seen.
        target = ((counts + 1) // 2).clamp(min=1).unsqueeze(1)
        reached = (valid.cumsum(1) >= target) & valid
        fallback = torch.full((n,), t // 2, device=x.device, dtype=torch.long)
        return torch.where(counts > 0, reached.float().argmax(1), fallback)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.middle_observed_frame(x)
        return self.net(x[torch.arange(x.shape[0], device=x.device), idx])
