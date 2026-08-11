"""Config-driven multimodal fusion condition network that returns an UNPOOLED
[B, N_condition, D] token sequence, for a DiT (DiTCrossAttn1d) that
cross-attends to condition tokens instead of consuming a single pooled AdaLN
vector.

Every ReLoc3R arm is built from this SAME class by varying which sensors are
listed in the YAML `sensors:` dict -- no code branching per variant, e.g.
  reloc3r_relfeat_only   : C = [Z_wheel, Z_dec1, Z_dec2]
  reloc3r_posthead_only  : C = [Z_wheel, Z_head1, Z_head2]
  reloc3r_goal_pool_*    : C = [Z_wheel, Z_goalpair]

Not implemented (explicitly out of scope per spec): confidence/learned
validity, NoMaD-style goal masking (the goal is ALWAYS attended -- no
goal_mask input exists here).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_condition.modality_encoders import build_encoder
from cleandiffuser.utils import SinusoidalEmbedding, Transformer


class TokenSequenceFusionCondition(BaseNNCondition):
    def __init__(
        self,
        sensors: Dict[str, dict],
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(sensors) == 0:
            raise ValueError("`sensors` spec is empty; enable at least one modality.")
        self.d_model = d_model
        self.sensor_names = list(sensors.keys())

        self.encoders = nn.ModuleDict()
        for name, spec in sensors.items():
            spec = dict(spec)
            spec.setdefault("name", name)
            self.encoders[name] = build_encoder(spec, d_model=d_model, nhead=nhead)

        self.n_modalities = len(self.sensor_names)
        self.modality_emb = nn.Parameter(torch.randn(self.n_modalities, d_model) * 0.02)
        self.token_pos_emb = SinusoidalEmbedding(d_model)
        self.tfm = Transformer(
            d_model, nhead, num_layers=num_layers,
            attn_dropout=dropout, ffn_dropout=dropout)

    def _gather_obs(self, name, condition):
        mask_key = f"{name}_valid_mask"
        if mask_key in condition:
            return {"data": condition[name], "valid_mask": condition[mask_key]}
        return condition[name]

    def forward(self, condition: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns tokens [B, N_condition, D] (single tensor -- the cleandiffuser
        backbone wrappers, e.g. ContinuousRectifiedFlow.loss/.sample, do
        `condition = self.model["condition"](condition)` and later
        `torch.cat([condition, torch.zeros_like(condition)], 0)` for CFG, so
        this must stay a plain tensor, not a tuple). Padding (missing/repeated
        history frames) is masked OUT of this fusion Transformer's own
        self-attention internally (a padded RGB-history frame cannot corrupt
        other tokens' fused representations) -- that satisfies "padding mask
        reaches attention" for the fusion stage. It is intentionally NOT
        re-exposed as a second return value for the downstream DiT
        cross-attention, to keep this class plug-compatible with the existing
        single-tensor condition-net interface; the padded positions that
        survive into `fused` are already context-mixed by masked self-attention
        (not raw duplicated pixels), so a few extra redundant tokens reaching
        DiT cross-attention is a minor residual imprecision, not a correctness
        bug. `mask` (branch-level CFG dropout) is accepted but unused -- this
        class has no readout token / no NoMaD masking."""
        missing = [n for n in self.sensor_names if n not in condition]
        if missing:
            raise KeyError(
                f"Missing condition keys for sensors {missing}. "
                f"Provided: {sorted(condition.keys())}")

        ref = self._gather_obs(self.sensor_names[0], condition)
        ref_t = ref["data"] if isinstance(ref, dict) else ref
        b, device = ref_t.shape[0], ref_t.device

        token_list, valid_list = [], []
        for i, name in enumerate(self.sensor_names):
            enc = self.encoders[name]
            obs = self._gather_obs(name, condition)
            tok = enc(obs) + self.modality_emb[i].view(1, 1, self.d_model)
            token_list.append(tok)
            n_tok = tok.shape[1]
            tvm = getattr(enc, "_token_valid_mask", None)
            if tvm is None:
                tvm = torch.ones(b, n_tok, dtype=torch.bool, device=device)
            valid_list.append(tvm)

        all_tokens = torch.cat(token_list, dim=1)              # [B, N, D]
        key_valid = torch.cat(valid_list, dim=1)                # [B, N] bool
        idx = torch.arange(all_tokens.shape[1], device=device)
        all_tokens = all_tokens + self.token_pos_emb(idx).unsqueeze(0)

        attn_mask = None
        if not key_valid.all():
            n = key_valid.shape[1]
            attn_mask = key_valid.float().unsqueeze(1).expand(b, n, n)  # (b,i,j), 0=masked key j
        fused, _ = self.tfm(all_tokens, mask=attn_mask)
        return fused
