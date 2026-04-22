import torch
import torch.nn as nn


class RawPatchEncoder(nn.Module):
    """
    Maps RGB frames [B, 3, 240, 320] to a 14x14 grid of 768-dim tokens [B, 196, 768]
    to plug into the same path as DINO patch features.
    Input is expected in [0, 1] float (as from DockingDataset).
    """

    def __init__(self, in_ch: int = 3, out_dim: int = 768, spatial: int = 14):
        super().__init__()
        self.spatial = spatial
        self.out_dim = out_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((spatial, spatial))
        self.proj = nn.Conv2d(256, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.pool(h)
        h = self.proj(h)
        return h.flatten(2).transpose(1, 2)
