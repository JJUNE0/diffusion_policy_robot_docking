"""Vision-based precision-docking model (ICP-distillation student).

Direction change (2026-06-23, user): instead of running ICP at runtime, learn a
*vision* model that does the mm precision phase, supervised offline by the ICP
labels we already generated (scripts/label_subgoals.py). At inference only vision
runs (RGB + LiDAR-BEV image, optional monocular depth) — no runtime algorithm.

Outputs:
  * pose  = [x_norm, y_norm, sin(theta), cos(theta)]  (dock pose in sensor frame;
            x,y normalized by dataset stats). The teacher is the ICP pose.
  * logvar = per-sample heteroscedastic log-variance = an **anomaly / confidence**
            signal (high => uncertain/anomalous during precision docking).

Kept dependency-light (torch only) and offline-safe (no pretrained download). A
pretrained RGB backbone or a monocular-depth branch can be swapped in later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallConvEncoder(nn.Module):
    """Lightweight conv encoder -> global feature vector (no pretrained weights)."""

    def __init__(self, in_ch: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class PrecisionDockingNet(nn.Module):
    """RGB(room1/2) + LiDAR-BEV (+optional depth) -> dock pose + anomaly."""

    def __init__(self, feat: int = 128, use_depth: bool = False):
        super().__init__()
        self.use_depth = use_depth
        self.rgb_enc = SmallConvEncoder(3, feat)        # shared across the two rooms
        self.bev_enc = SmallConvEncoder(1, feat)
        if use_depth:
            self.depth_enc = SmallConvEncoder(1, feat)
        n_branch = 3 + (1 if use_depth else 0)
        self.fuse = nn.Sequential(
            nn.Linear(feat * n_branch, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.pose_head = nn.Linear(256, 4)              # x, y, sin, cos
        self.logvar_head = nn.Linear(256, 1)            # anomaly / confidence

    def forward(self, room1, room2, bev, depth=None):
        feats = [self.rgb_enc(room1), self.rgb_enc(room2), self.bev_enc(bev)]
        if self.use_depth and depth is not None:
            feats.append(self.depth_enc(depth))
        h = self.fuse(torch.cat(feats, dim=1))
        raw = self.pose_head(h)
        xy = raw[:, :2]
        sincos = F.normalize(raw[:, 2:4], dim=1)        # keep (sin,cos) on the unit circle
        pose = torch.cat([xy, sincos], dim=1)
        logvar = self.logvar_head(h).squeeze(1)
        return pose, logvar


def pose_nll_loss(pose: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL over the 4-dim pose (shared per-sample variance).

    L = 0.5 * exp(-s) * ||pose - target||^2 + 0.5 * D * s   (s = logvar, D = 4)
    The variance head therefore learns to be large where the pose is hard to read
    from vision — exactly the precision-docking anomaly/confidence signal.
    """
    se = ((pose - target) ** 2).sum(dim=1)
    return (0.5 * torch.exp(-logvar) * se + 0.5 * 4 * logvar).mean()
