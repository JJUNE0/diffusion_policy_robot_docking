import torch
import torch.nn as nn


class LidarMapEncoder(nn.Module):
    """
    BEV occupancy map -> patch tokens.

    Maps an egocentric BEV occupancy map [B, C, H, W] (typically C=2: occupied + log_density)
    into a (spatial x spatial) grid of `out_dim`-dim tokens [B, spatial*spatial, out_dim],
    so the lidar branch can be plugged into the same fusion path used by the camera branches
    in `SensorFusionConditionNetwork` (which expects 196 = 14x14 tokens of dim 768 by default).

    The architecture mirrors `vision/raw_patch_encoder.py` but starts from a small
    spatial map (e.g. 120x120) and uses GroupNorm + GELU.

    Input is expected as float in [0, 1] (the dataset divides uint8 BEV maps by 255).
    """

    def __init__(self, in_ch: int = 2, out_dim: int = 768, spatial: int = 14):
        super().__init__()
        self.in_ch = in_ch
        self.spatial = spatial
        self.out_dim = out_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((spatial, spatial))
        self.proj = nn.Conv2d(256, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] float in [0, 1].

        Returns:
            [B, spatial*spatial, out_dim] patch tokens.
        """
        h = self.stem(x)
        h = self.pool(h)
        h = self.proj(h)
        return h.flatten(2).transpose(1, 2)
