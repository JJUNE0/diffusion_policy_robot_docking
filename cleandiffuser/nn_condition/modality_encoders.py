"""Modality encoder registry for config-driven sensor fusion.

Each modality (the wheel encoder, a ReLoc3R relational stream, ...) is encoded
by a small nn.Module mapping its raw observation to transformer tokens
[B, N_tokens, d_model]. Encoders are looked up by the string `encoder` field in
the YAML `sensors:` spec, so adding/removing a modality for an ablation is just
editing that dict -- no code changes in the dataset, condition net, or train
loop. See TokenSequenceFusionCondition and utils/modular_dataset.py.
"""
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

ENCODER_REGISTRY: Dict[str, type] = {}


class PerceiverResampler(nn.Module):
    """Fixed-size latent bottleneck over a variable-length token set."""

    def __init__(self, d_model: int, num_latents: int = 16, nhead: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N_patch, d_model]
            cond: optional conditioning tensor broadcastable to [B, N_latent, d_model]
            key_padding_mask: [B, N_patch] bool, True = ignore

        Returns:
            [B, N_latent, d_model]
        """
        b = x.shape[0]
        q = self.latents.repeat(b, 1, 1)

        if cond is not None:
            q = q + cond

        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm,
                                      key_padding_mask=key_padding_mask)

        out = q + attn_out
        out = out + self.ffn(self.ln_post(out))
        return out


def _unpack(obs):
    """History-mode encoders may receive either a raw tensor or {"data":...,
    "valid_mask": [B,T] bool, True=real frame} (TokenSequenceFusionCondition,
    which needs per-token padding masks to reach attention -- 2026-07-25
    acceptance criterion). Returns (data, valid_mask_or_None)."""
    if isinstance(obs, dict) and "data" in obs:
        return obs["data"], obs.get("valid_mask")
    return obs, None


def register_encoder(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in ENCODER_REGISTRY:
            raise KeyError(f"Modality encoder '{name}' already registered.")
        ENCODER_REGISTRY[name] = cls
        cls.encoder_name = name
        return cls
    return deco


def build_encoder(spec: dict, d_model: int, nhead: int) -> "BaseModalityEncoder":
    kind = spec["encoder"]
    if kind not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown modality encoder '{kind}'. Registered: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[kind](spec, d_model=d_model, nhead=nhead)


class BaseModalityEncoder(nn.Module):
    """Maps one modality's observation to tokens [B, N, d_model].

    Subclasses implement `encode(obs) -> [B, N, d_model]`. Any temporal/slot
    embedding is the encoder's own responsibility; the parent fusion net adds a
    single per-modality embedding on top (so encoders stay modality-agnostic).
    """

    encoder_name: str = "base"

    def __init__(self, spec: dict, d_model: int, nhead: int):
        super().__init__()
        self.spec = spec
        self.name = spec.get("name", self.encoder_name)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_prob = float(spec.get("dropout_prob", 0.0))
        self._token_valid_mask = None  # [B, N_tokens] bool, True=real; set by
        # encode() when the caller supplies a per-frame valid_mask (see
        # _unpack above). None means "caller doesn't track padding" -> treat
        # as fully valid (matches every pre-2026-07-25 call site).

    def obs_keys(self):
        """Condition-dict keys this encoder consumes (for documentation/validation)."""
        return [self.name]

    def branch_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """NoMaD-style whole-branch dropout for classifier-free guidance (train only)."""
        if (not self.training) or self.dropout_prob <= 0.0:
            return tokens
        b = tokens.shape[0]
        keep = (torch.rand(b, device=tokens.device) > self.dropout_prob).float().view(b, 1, 1)
        return tokens * keep

    def encode(self, obs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs) -> torch.Tensor:
        return self.branch_dropout(self.encode(obs))


@register_encoder("motion")
class MotionEncoder(BaseModalityEncoder):
    """Low-dim time series (wheel encoder velocity, IMU, command, ...) -> per-step
    tokens. obs: [B, T, dim]. Dimension-agnostic: `encoder`, `imu`, `command` all
    reuse this by setting `dim`/`horizon`/`source` in the spec. Produces T tokens.
    """

    def __init__(self, spec, d_model, nhead):
        super().__init__(spec, d_model, nhead)
        self.dim = int(spec["dim"])
        self.horizon = int(spec["horizon"])
        self.proj = nn.Sequential(
            nn.Linear(self.dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.time_emb = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.n_tokens = self.horizon

    def encode(self, obs):
        x, valid_mask = _unpack(obs)
        x = x.float()
        b, t, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"[{self.name}] expected dim={self.dim}, got {dim}.")
        if t != self.horizon:
            raise ValueError(f"[{self.name}] expected horizon={self.horizon}, got {t}.")
        self._token_valid_mask = valid_mask  # 1 token/step -> mask is already token-shaped
        return self.proj(x) + self.time_emb.view(1, t, self.d_model)


@register_encoder("reloc3r_relation")
class Reloc3rRelationEncoder(BaseModalityEncoder):
    """Reloc3r decoder's cross-attended patch tokens (POST cross-attention,
    PRE pose-head pooling) -> Perceiver latents. These tokens have already
    cross-attended into the goal frame's (or the current frame's) stream inside
    Reloc3r's own decoder (arxiv 2412.08376; see Reloc3rRelpose._decoder,
    reloc3r/reloc3r/reloc3r_relpose.py). obs: [B, Tv, n_patch, feat_dim], e.g.
    [B, Tv, 196, 768] for the dec1/dec2 streams precomputed by
    scripts/precompute_reloc3r_dec_features.py, or [B, Tv, 1, 1024] for the
    post-pose-head taps from scripts/precompute_reloc3r_head_features.py.
    Serves BOTH the dec1 ("goal-aware history": current frame's stream after
    attending into goal) and dec2 ("current-aware goal") sensor entries -- two
    independently-weighted instances of this SAME class, differentiated purely
    by which `source`/`file` the YAML sensor entry points at; no
    dec1-vs-dec2-specific logic lives in Python. Produces Tv * num_latents
    tokens; each sensor owns its feat_dim->d_model projection.
    """

    def __init__(self, spec, d_model, nhead):
        spec = dict(spec)
        spec.setdefault("feat_dim", 768)
        super().__init__(spec, d_model, nhead)
        self.horizon = int(spec["horizon"])            # vision horizon (post-stride)
        self.num_latents = int(spec.get("num_latents", 16))
        self.n_patch = int(spec.get("n_patch", 196))
        self.feat_dim = int(spec.get("feat_dim", 768))
        self.vision_proj = nn.Linear(self.feat_dim, d_model)
        self.resampler = PerceiverResampler(d_model, num_latents=self.num_latents, nhead=nhead)
        self.time_emb = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(self.num_latents, d_model) * 0.02)
        self.n_tokens = self.horizon * self.num_latents

    def encode(self, obs):
        feat, valid_mask = _unpack(obs)
        feat = feat.float()
        b, t, n_patch, feat_dim = feat.shape
        if t != self.horizon:
            raise ValueError(f"[{self.name}] expected vision horizon={self.horizon}, got {t}.")
        if n_patch != self.n_patch or feat_dim != self.feat_dim:
            raise ValueError(
                f"[{self.name}] expected [*,{self.n_patch},{self.feat_dim}], got [*,{n_patch},{feat_dim}].")
        flat = feat.reshape(b * t, n_patch, feat_dim)
        lat = self.resampler(self.vision_proj(flat))            # [B*T, n_lat, d]
        lat = lat.view(b, t, self.num_latents, self.d_model)
        lat = (lat
               + self.time_emb.view(1, t, 1, self.d_model)
               + self.slot_emb.view(1, 1, self.num_latents, self.d_model))
        if valid_mask is not None:
            # one frame-level flag -> num_latents identical token-level flags
            self._token_valid_mask = valid_mask.unsqueeze(-1).expand(b, t, self.num_latents).reshape(
                b, t * self.num_latents)
        else:
            self._token_valid_mask = None
        return lat.reshape(b, t * self.num_latents, self.d_model)


@register_encoder("reloc3r_goal_pair")
class Reloc3rGoalPairEncoder(BaseModalityEncoder):
    """Random goal-pool pair -> exact frozen ReLoc3R decoder -> policy tokens.

    Input is [B,T,2,196,1024], where pair axis 0 is a cached history-frame
    ViT-L encoder feature and axis 1 is the sampled goal's encoder feature.
    The frozen 12-layer ReLoc3R decoder produces its two 768-D relational
    streams online; only the projections, Perceiver resamplers and embeddings
    below are trainable/saved with the diffusion policy.
    """

    def __init__(self, spec, d_model, nhead):
        super().__init__(spec, d_model, nhead)
        self.horizon = int(spec["horizon"])
        self.n_patch = int(spec.get("n_patch", 196))
        self.feat_dim = int(spec.get("feat_dim", 1024))
        self.dec_dim = int(spec.get("dec_dim", 768))
        self.num_latents = int(spec.get("num_latents", 16))
        self.checkpoint = str(spec.get("reloc3r_checkpoint", "siyan824/reloc3r-224"))
        self.decoder_chunk = int(spec.get("decoder_chunk", 4))
        if self.decoder_chunk < 1:
            raise ValueError(f"[{self.name}] decoder_chunk must be >=1")

        self.proj1 = nn.Linear(self.dec_dim, d_model)
        self.proj2 = nn.Linear(self.dec_dim, d_model)
        self.resampler1 = PerceiverResampler(
            d_model, num_latents=self.num_latents, nhead=nhead)
        self.resampler2 = PerceiverResampler(
            d_model, num_latents=self.num_latents, nhead=nhead)
        self.time_emb1 = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.time_emb2 = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.slot_emb1 = nn.Parameter(torch.randn(self.num_latents, d_model) * 0.02)
        self.slot_emb2 = nn.Parameter(torch.randn(self.num_latents, d_model) * 0.02)
        self.stream_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.n_tokens = 2 * self.horizon * self.num_latents

    def encode(self, obs):
        pair, valid_mask = _unpack(obs)
        # Keep the multi-GiB pair tensor compact under H100 BF16 autocast.
        # In FP32 mode retain the previous exact behavior.
        pair = pair.to(torch.get_autocast_dtype("cuda")) if (
            pair.is_cuda and torch.is_autocast_enabled("cuda")
        ) else pair.float()
        if pair.dim() != 5:
            raise ValueError(
                f"[{self.name}] expected [B,T,2,P,D], got {tuple(pair.shape)}")
        b, t, two, n_patch, feat_dim = pair.shape
        if (t, two, n_patch, feat_dim) != (
            self.horizon, 2, self.n_patch, self.feat_dim
        ):
            raise ValueError(
                f"[{self.name}] expected [B,{self.horizon},2,{self.n_patch},"
                f"{self.feat_dim}], got {tuple(pair.shape)}"
            )

        from cleandiffuser.nn_condition.reloc3r_goal_pair import (
            get_frozen_reloc3r_decoder,
        )
        service = get_frozen_reloc3r_decoder(self.checkpoint, pair.device)
        current = pair[:, :, 0].reshape(b * t, n_patch, feat_dim)
        goal = pair[:, :, 1].reshape(b * t, n_patch, feat_dim)
        dec1, dec2 = service.decode(current, goal, self.decoder_chunk)

        lat1 = self.resampler1(self.proj1(dec1)).view(
            b, t, self.num_latents, self.d_model)
        lat2 = self.resampler2(self.proj2(dec2)).view(
            b, t, self.num_latents, self.d_model)
        lat1 = (
            lat1
            + self.time_emb1.view(1, t, 1, self.d_model)
            + self.slot_emb1.view(1, 1, self.num_latents, self.d_model)
            + self.stream_emb[0].view(1, 1, 1, self.d_model)
        )
        lat2 = (
            lat2
            + self.time_emb2.view(1, t, 1, self.d_model)
            + self.slot_emb2.view(1, 1, self.num_latents, self.d_model)
            + self.stream_emb[1].view(1, 1, 1, self.d_model)
        )
        if valid_mask is not None:
            token_valid = valid_mask.unsqueeze(-1).expand(
                b, t, self.num_latents).reshape(b, t * self.num_latents)
            self._token_valid_mask = torch.cat([token_valid, token_valid], dim=1)
        else:
            self._token_valid_mask = None
        return torch.cat(
            [
                lat1.reshape(b, t * self.num_latents, self.d_model),
                lat2.reshape(b, t * self.num_latents, self.d_model),
            ],
            dim=1,
        )
