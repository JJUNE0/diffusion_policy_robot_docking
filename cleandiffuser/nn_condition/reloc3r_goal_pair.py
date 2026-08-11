"""Frozen ReLoc3R decoder service for random goal-pool pairing.

The service deliberately lives outside the trainable module tree:

* ReLoc3R weights are frozen preprocessing, not policy parameters.
* DiffusionModel deep-copies the trainable model for EMA. Registering the
  frozen decoder would duplicate it in GPU memory and every checkpoint.
* A process-local singleton is shared by the online and EMA condition nets.

Training checkpoints therefore retain the dependency on
``siyan824/reloc3r-224`` (the same dependency as cache generation).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

_SERVICES: Dict[Tuple[str, str], "FrozenReloc3rDecoderService"] = {}


class FrozenReloc3rDecoderService:
    def __init__(self, checkpoint: str, device: torch.device):
        repo = Path(__file__).resolve().parents[2]
        outer = str(repo / "reloc3r")
        if outer not in sys.path:
            sys.path.insert(0, outer)
        from reloc3r.reloc3r_relpose import Reloc3rRelpose

        self.model = Reloc3rRelpose.from_pretrained(checkpoint).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device
        with torch.no_grad():
            shape = torch.tensor([[224, 224]], dtype=torch.int64, device=device)
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            _token, pos = self.model.patch_embed(dummy, true_shape=shape)
            self.pos = pos.detach()

    @torch.no_grad()
    def decode(self, current: torch.Tensor, goal: torch.Tensor, chunk: int):
        """Return final normalized decoder streams for [M,196,1024] pairs."""
        out1, out2 = [], []
        for start in range(0, current.shape[0], chunk):
            stop = min(start + chunk, current.shape[0])
            f1 = current[start:stop].to(self.device, non_blocking=True)
            f2 = goal[start:stop].to(self.device, non_blocking=True)
            pos = self.pos.expand(stop - start, -1, -1)
            d1, d2 = self.model._decoder(f1, pos, f2, pos)
            out1.append(d1[-1])
            out2.append(d2[-1])
        return torch.cat(out1, dim=0), torch.cat(out2, dim=0)


def get_frozen_reloc3r_decoder(
    checkpoint: str, device: torch.device
) -> FrozenReloc3rDecoderService:
    key = (checkpoint, str(device))
    service = _SERVICES.get(key)
    if service is None:
        service = FrozenReloc3rDecoderService(checkpoint, device)
        _SERVICES[key] = service
    return service


def clear_frozen_reloc3r_decoders():
    """Test/worker cleanup hook."""
    _SERVICES.clear()
