"""Vision-based precision-docking model (ICP-distillation student).

At inference: RGB(room1/2) + LiDAR-BEV (+optional depth) -> dock pose + anomaly,
with NO runtime algorithm (ICP is only the offline teacher). See model.py /
dataset.py and scripts/train_precision_vision.py.
"""

from .dataset import PrecisionDockingFrameDataset, points_to_bev
from .model import PrecisionDockingNet, SmallConvEncoder, pose_nll_loss

__all__ = [
    "PrecisionDockingNet",
    "SmallConvEncoder",
    "pose_nll_loss",
    "PrecisionDockingFrameDataset",
    "points_to_bev",
]
