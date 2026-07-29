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


def _unpack(obs):
    """History-mode encoders may receive either a raw tensor (legacy call
    sites, e.g. ModularSensorFusionCondition) or {"data":..., "valid_mask":
    [B,T] bool, True=real frame} (TokenSequenceFusionCondition, which needs
    per-token padding masks to reach attention -- 2026-07-25 acceptance
    criterion). Returns (data, valid_mask_or_None)."""
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


@register_encoder("reloc3r_relation")
class Reloc3rRelationEncoder(DinoImageEncoder):
    """Reloc3r decoder's cross-attended patch tokens (POST cross-attention,
    PRE pose-head pooling) -> Perceiver latents. Unlike Reloc3rImageEncoder
    (raw ViT-L *encoder* patch features, independently encoded per frame),
    these tokens have already cross-attended into the goal frame's (or the
    current frame's) stream inside Reloc3r's own decoder (arxiv 2412.08376;
    see Reloc3rRelpose._decoder, reloc3r/reloc3r/reloc3r_relpose.py). obs:
    [B, Tv, 196, 768] (dec_embed_dim=768, precomputed by
    scripts/precompute_reloc3r_dec_features.py). Serves BOTH the dec1
    ("goal-aware history": current frame's stream after attending into goal)
    and dec2 ("current-aware goal": the same stream rotation/geometry already
    collapse to a point estimate, kept here at full patch resolution) sensor
    entries -- two independently-weighted instances of this SAME class,
    differentiated purely by which `source`/`file` the YAML sensor entry
    points at; no dec1-vs-dec2-specific logic lives in Python. Kept as its
    own registry name (not reusing `reloc3r_image`) so this backbone-vs-
    relational distinction is a one-word config change.
    """

    def __init__(self, spec, d_model, nhead):
        spec = dict(spec)
        spec.setdefault("feat_dim", 768)
        super().__init__(spec, d_model, nhead)


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
        pair = pair.float()
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


@register_encoder("goal_image")
class GoalImageEncoder(BaseModalityEncoder):
    """STATIC goal frame as DINO/Reloc3r patch features -> Perceiver latents,
    ONE token set (no time dimension -- a goal is a single frame, not a
    history). obs: [B, n_patch, feat_dim] (mode="current" in the dataset spec,
    e.g. source: dino_bottom, or reloc3r_bottom for the R-Geo arm's appearance
    branch). No NoMaD-style masking (docs/0725_reloc3r_test/reloc3r/
    reloc3r_0725.md: "Nomad-style goal mask 제거해야함 (goal이 항상 제공되도록 할
    것)") -- the goal is always attended to; there is no goal_mask input here
    at all, unlike the legacy SensorFusionConditionNetwork's NoMaD branch.
    Always fully valid (a static goal frame is never "missing/padded").
    """

    def __init__(self, spec, d_model, nhead):
        super().__init__(spec, d_model, nhead)
        self.num_latents = int(spec.get("num_latents", 16))
        self.n_patch = int(spec.get("n_patch", 196))
        self.feat_dim = int(spec.get("feat_dim", 768))
        self.proj = nn.Linear(self.feat_dim, d_model)
        self.resampler = PerceiverResampler(d_model, num_latents=self.num_latents, nhead=nhead)
        self.slot_emb = nn.Parameter(torch.randn(self.num_latents, d_model) * 0.02)
        self.n_tokens = self.num_latents

    def encode(self, obs):
        feat, _ = _unpack(obs)
        feat = feat.float()
        b, n_patch, feat_dim = feat.shape
        if n_patch != self.n_patch or feat_dim != self.feat_dim:
            raise ValueError(
                f"[{self.name}] expected [*,{self.n_patch},{self.feat_dim}], got [*,{n_patch},{feat_dim}].")
        lat = self.resampler(self.proj(feat))  # [B, n_lat, d]
        self._token_valid_mask = None  # static goal frame: always fully valid
        return lat + self.slot_emb.view(1, self.num_latents, self.d_model)


@register_encoder("geometry")
class GeometryTokenEncoder(BaseModalityEncoder):
    """Cached Reloc3r body-frame geometry vector -> ONE token (docs/
    0725_reloc3r_test/reloc3r/reloc3r_0725.md: "Reloc3r geometry는 하나의
    geometry token으로 사용한다"). obs: [B, 4] = [dx_body, dy_body,
    sin(relative_yaw_body), cos(relative_yaw_body)] (mode="current", source:
    geometry_bottom, see scripts/build_reloc3r_geometry_cache.py). Frozen
    Reloc3r never runs in the training loop -- this is a plain small-MLP
    lookup of an offline-cached vector, same spirit as MotionEncoder but for a
    single static value (no time dimension) rather than a history.
    """

    def __init__(self, spec, d_model, nhead):
        super().__init__(spec, d_model, nhead)
        self.dim = int(spec.get("dim", 4))
        self.proj = nn.Sequential(
            nn.Linear(self.dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.n_tokens = 1

    def encode(self, obs):
        x, _ = _unpack(obs)
        x = x.float()
        if x.shape[-1] != self.dim:
            raise ValueError(f"[{self.name}] expected dim={self.dim}, got {x.shape[-1]}.")
        self._token_valid_mask = None  # always present (goal_appearance_geometry arm only)
        return self.proj(x).unsqueeze(1)  # [B, 1, D]


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
