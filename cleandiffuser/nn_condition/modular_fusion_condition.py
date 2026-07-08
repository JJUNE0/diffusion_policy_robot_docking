"""Config-driven multimodal fusion condition network.

Reads a `sensors` spec (ordered dict: name -> spec) and builds one encoder per
sensor from the modality registry. All sensor tokens are concatenated with a
readout token, each tagged by a learned per-modality embedding, fused by a
Transformer; the readout token is returned as the [B, d_model] condition vector.
Ablations are done purely by editing the `sensors` dict -- no code changes here
or in the dataset/train loop.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_condition.modality_encoders import PointCloudEncoder, build_encoder
from cleandiffuser.nn_condition.sensor_fusion_condition import CrossAttnPoseHead
from cleandiffuser.utils import SinusoidalEmbedding, Transformer


class ModularSensorFusionCondition(BaseNNCondition):
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

        # One learned embedding per modality + one for the readout token, indexed
        # by enumeration order of `sensors` (stable given a fixed spec).
        self.n_modalities = len(self.sensor_names)
        self.modality_emb = nn.Parameter(torch.randn(self.n_modalities + 1, d_model) * 0.02)
        self.readout_idx = self.n_modalities

        self.readout_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_pos_emb = SinusoidalEmbedding(d_model)
        self.tfm = Transformer(
            d_model, nhead, num_layers=num_layers,
            attn_dropout=dropout, ffn_dropout=dropout)

        # Auxiliary precision head (config-driven, ablatable). A sensor declares
        # `head: aux_pose` to attach an ICP-distilled dock-pose head that predicts
        # [x_norm, y_norm, sin, cos]. When it sits on a `pointcloud` sensor the
        # head cross-attends to the RAW point tokens (mm precision, ICP-like);
        # otherwise it falls back to a pooled-readout MLP. Drop the `head` field
        # (or the sensor) to ablate it. train.py reads `_aux_pred` for the masked
        # distillation loss -- None when no head is configured.
        self.aux_sensor = None
        self.aux_head = None
        self.aux_head_kind = None
        for name, spec in sensors.items():
            head = spec.get("head")
            if head is None:
                continue
            if head != "aux_pose":
                raise ValueError(f"sensor '{name}': unknown head '{head}' (only 'aux_pose').")
            if self.aux_head is not None:
                raise ValueError("at most one `aux_pose` head is supported.")
            self.aux_sensor = name
            if isinstance(self.encoders[name], PointCloudEncoder):
                self.aux_head = CrossAttnPoseHead(d_model, nhead)   # raw-point cross-attn
                self.aux_head_kind = "cross_attn"
            else:
                self.aux_head = nn.Sequential(                      # pooled fallback
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 4))
                self.aux_head_kind = "pooled"
        self._aux_pred = None

    def _gather_obs(self, name, encoder, condition):
        if isinstance(encoder, PointCloudEncoder):
            return {"points": condition[name], "npoints": condition[f"{name}_npoints"]}
        return condition[name]

    def forward(self, condition: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        missing = [n for n in self.sensor_names if n not in condition]
        if missing:
            raise KeyError(
                f"Missing condition keys for sensors {missing}. "
                f"Provided: {sorted(condition.keys())}")

        first = self.encoders[self.sensor_names[0]]
        ref = self._gather_obs(self.sensor_names[0], first, condition)
        ref_t = ref["points"] if isinstance(ref, dict) else ref
        b, device = ref_t.shape[0], ref_t.device

        token_list = []
        for i, name in enumerate(self.sensor_names):
            enc = self.encoders[name]
            obs = self._gather_obs(name, enc, condition)
            tok = enc(obs) + self.modality_emb[i].view(1, 1, self.d_model)
            token_list.append(tok)

        readout = self.readout_emb.repeat(b, 1, 1) + \
            self.modality_emb[self.readout_idx].view(1, 1, self.d_model)
        token_list.append(readout)

        all_tokens = torch.cat(token_list, dim=1)
        idx = torch.arange(all_tokens.shape[1], device=device)
        all_tokens = all_tokens + self.token_pos_emb(idx).unsqueeze(0)

        fused = self.tfm(all_tokens)[0]
        out = fused[:, -1, :]

        # Aux dock-pose prediction (read by the train loop for the masked ICP
        # distillation loss). Cross-attn over raw points when the head sits on a
        # pointcloud sensor; else pooled-readout MLP.
        self._aux_pred = None
        if self.aux_head is not None:
            if self.aux_head_kind == "cross_attn":
                enc = self.encoders[self.aux_sensor]
                self._aux_pred = self.aux_head(enc._point_tokens, enc._point_mask, out)
            else:
                self._aux_pred = self.aux_head(out)

        if mask is not None:
            out = out * mask.view(b, 1).float()
        return out
