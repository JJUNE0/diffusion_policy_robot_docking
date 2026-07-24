"""Modality encoder registry for config-driven sensor fusion.

Each modality (a camera, the wheel encoder, IMU, LiDAR, ...) is encoded by a
small nn.Module mapping its raw observation to transformer tokens
[B, N_tokens, d_model]. Encoders are looked up by the string `encoder` field in
the YAML `sensors:` spec, so adding/removing a modality for an ablation is just
editing that dict -- no code changes in the dataset, condition net, or train
loop. See ModularSensorFusionCondition and dataset/modular_dataset.py.
"""
from typing import Callable, Dict

import torch
import torch.nn as nn

from cleandiffuser.nn_condition.sensor_fusion_condition import PerceiverResampler

ENCODER_REGISTRY: Dict[str, type] = {}


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
        x = obs.float()
        b, t, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"[{self.name}] expected dim={self.dim}, got {dim}.")
        if t != self.horizon:
            raise ValueError(f"[{self.name}] expected horizon={self.horizon}, got {t}.")
        return self.proj(x) + self.time_emb.view(1, t, self.d_model)


@register_encoder("imu")
class ImuEncoder(MotionEncoder):
    """IMU (gyro xyz + accel xyz = 6D) time series. Currently identical to the
    generic MotionEncoder (per-step MLP + temporal embedding). Kept as its own
    registry name so a smarter IMU encoding (integrate to orientation, temporal
    conv over the high-rate stream, gravity removal, ...) can be swapped in later
    WITHOUT touching any config that already says `encoder: imu`.
    TODO(imu): decide the right representation; 6-axis raw is the placeholder.
    """
    pass


@register_encoder("rotation")
class RotationEncoder(MotionEncoder):
    """Reloc3r rotation-to-goal signal as a low-dim time series. obs: [B, T, 7]
    where the 7 dims are [R[:,0] (3), R[:,1] (3), geodesic_angle (1)] per frame
    (see scripts/precompute_reloc3r_cache.py). Validated as the reliable Reloc3r
    channel in docs/reloc3r.md (heading r=0.93, strong at close range); the
    translation/direction channel is deliberately excluded (lidar owns it).
    Identical mechanics to MotionEncoder (per-step MLP + temporal embedding);
    kept as its own registry name so a smarter rotation encoding can be swapped
    in later without touching any config that says `encoder: rotation`.
    """
    pass


@register_encoder("direction")
class DirectionEncoder(MotionEncoder):
    """Reloc3r translation-DIRECTION-to-goal signal as a low-dim time series.
    obs: [B, T, 3], the unit translation vector (scale-ambiguous by construction
    -- see docs/reloc3r.md) per frame (scripts/precompute_reloc3r_direction.py).
    docs/reloc3r.md found this channel degrades sharply at close range (median
    16.9deg error at 0-0.3m vs 2.6deg at 0.9-1.1m) -- excluded from the R+rot
    arm by default; this encoder exists so the rot-vs-rot+dir ablation can add
    it back in without touching any other code, per user request 2026-07-22.
    Identical mechanics to MotionEncoder (per-step MLP + temporal embedding).
    """
    pass


@register_encoder("dino_image")
class DinoImageEncoder(BaseModalityEncoder):
    """Camera view as DINO patch features -> Perceiver latents.
    obs: [B, Tv, n_patch, feat_dim] (precomputed DINO features). Produces
    Tv * num_latents tokens. Each camera owns its DINO->d_model projection.
    """

    def __init__(self, spec, d_model, nhead):
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
        feat = obs.float()
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
        return lat.reshape(b, t * self.num_latents, self.d_model)


@register_encoder("reloc3r_image")
class Reloc3rImageEncoder(DinoImageEncoder):
    """Camera view as Reloc3r ViT-L encoder patch features -> Perceiver latents.
    Identical to DinoImageEncoder but defaults feat_dim to 1024 (ViT-L) instead
    of 768 (DINOv3 ViT-B). obs: [B, Tv, 196, 1024] (precomputed, see
    scripts/precompute_reloc3r_cache.py). Kept as its own registry name so the
    DINO-vs-Reloc3r backbone swap is a one-word config change.
    """

    def __init__(self, spec, d_model, nhead):
        spec = dict(spec)
        spec.setdefault("feat_dim", 1024)
        super().__init__(spec, d_model, nhead)


@register_encoder("pointcloud")
class PointCloudEncoder(BaseModalityEncoder):
    """Raw 2D LiDAR point set -> masked Perceiver latents.
    obs: dict {"points": [B, M, 2] (zero-padded), "npoints": [B]}. Produces
    num_latents tokens; padded points are masked out of the cross-attention.
    """

    def __init__(self, spec, d_model, nhead):
        super().__init__(spec, d_model, nhead)
        self.point_dim = int(spec.get("point_dim", 2))
        self.num_latents = int(spec.get("num_latents", 16))
        self.proj = nn.Sequential(
            nn.Linear(self.point_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.resampler = PerceiverResampler(d_model, num_latents=self.num_latents, nhead=nhead)
        self.slot_emb = nn.Parameter(torch.randn(self.num_latents, d_model) * 0.02)
        self.n_tokens = self.num_latents
        self._point_tokens = None      # exposed for an optional cross-attn aux head
        self._point_mask = None

    def obs_keys(self):
        return [self.name, f"{self.name}_npoints"]

    def encode(self, obs):
        points = obs["points"].float()
        npoints = obs["npoints"]
        b, m, _ = points.shape
        tok = self.proj(points)
        valid = torch.arange(m, device=points.device)[None, :] < npoints.view(-1, 1)
        pad_mask = ~valid                                 # True = ignore
        pad_mask[npoints <= 0, 0] = False                 # avoid all-masked NaN
        self._point_tokens = tok
        self._point_mask = pad_mask
        lat = self.resampler(tok, key_padding_mask=pad_mask)
        return lat + self.slot_emb.view(1, self.num_latents, self.d_model)
