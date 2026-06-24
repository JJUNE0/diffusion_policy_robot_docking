from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import SinusoidalEmbedding, Transformer


class PerceiverResampler(nn.Module):
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
            key_padding_mask: [B, N_patch] bool, True = ignore (for padded point sets)

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


class CrossAttnPoseHead(nn.Module):
    """Dock-pose head that cross-attends a learned pose query to the RAW LiDAR
    point tokens (direct geometric access) instead of regressing from the pooled
    readout vector. The pooled-MLP head plateaued at ~3 cm because the precise
    point geometry is diluted in one global vector; here the query reasons over
    individual points (ICP-like), enabling mm precision.

    Output: [x_norm, y_norm, sin(theta), cos(theta)].
    """

    def __init__(self, d_model: int, nhead: int = 6):
        super().__init__()
        self.pose_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_post = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.out = nn.Linear(d_model, 4)

    def forward(self, point_tokens, key_padding_mask, cond_vec=None):
        """point_tokens [B,M,d], key_padding_mask [B,M] (True=ignore), cond_vec [B,d] opt."""
        b = point_tokens.shape[0]
        q = self.pose_query.repeat(b, 1, 1)
        if cond_vec is not None:                       # add fused global context
            q = q + cond_vec.unsqueeze(1)
        a, _ = self.attn(self.ln_q(q), self.ln_kv(point_tokens), self.ln_kv(point_tokens),
                         key_padding_mask=key_padding_mask)
        h = q + a
        h = h + self.ffn(self.ln_post(h))
        raw = self.out(h.squeeze(1))
        xy = raw[:, :2]
        sincos = F.normalize(raw[:, 2:4], dim=1)       # keep (sin,cos) on the unit circle
        return torch.cat([xy, sincos], dim=1)


class SensorFusionConditionNetwork(BaseNNCondition):
    """
    Two-camera vision + velocity condition encoder.

    Expected condition dict:
        - dino_feat1: [B, Tv, 196, 768]
        - dino_feat2: [B, Tv, 196, 768]
        - velocity:   [B, Tm, 2]   = normalized [v, w]

    where
        - Tv = vision_horizon (default: 5)
        - Tm = obs_horizon    (default: 30)

    Output:
        - [B, d_model] condition vector from the readout token.
    """

    def __init__(
        self,
        state_dim: int = 2,
        obs_horizon: int = 30,
        vision_horizon: int = 5,
        d_model: int = 768,
        nhead: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_image_latents: int = 16,
        velocity_dim: int = 2,
        velocity_dropout_prob: float = 0.2,
        use_goal: bool = False,
        num_goal_latents: int = 16,
        use_lidar_points: bool = False,
        num_lidar_latents: int = 16,
        use_aux_pose: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.vision_horizon = vision_horizon
        self.d_model = d_model
        self.dropout = dropout
        self.num_image_latents = num_image_latents
        self.velocity_dim = velocity_dim
        self.velocity_dropout_prob = velocity_dropout_prob
        self.use_goal = use_goal
        self.num_goal_latents = num_goal_latents

        # Vision branches: one branch per room.
        self.vision_proj = nn.Linear(768, d_model)
        self.room1_resampler = PerceiverResampler(d_model, num_latents=num_image_latents, nhead=nhead)
        self.room2_resampler = PerceiverResampler(d_model, num_latents=num_image_latents, nhead=nhead)

        # Motion branch.
        self.velocity_proj = nn.Sequential(
            nn.Linear(velocity_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Separate temporal embeddings because vision and velocity have different horizons.
        self.vision_time_emb = nn.Parameter(torch.randn(vision_horizon, d_model) * 0.02)
        self.motion_time_emb = nn.Parameter(torch.randn(obs_horizon, d_model) * 0.02)
        self.image_slot_emb = nn.Parameter(torch.randn(num_image_latents, d_model) * 0.02)

        # modality indices: 0=image1, 1=image2, 2=velocity, 3=readout
        self.modality_emb = nn.Parameter(torch.randn(4, d_model) * 0.02)

        # Goal-feature conditioning (CLAUDE.md §2.3 Loss A). The goal is the DINO
        # feature of a future/docked camera frame; NoMaD-style masking lets one
        # policy learn both goal-conditioned and undirected docking. Reuses
        # vision_proj (same DINO space) but has its own resamplers/embeddings so
        # the goal is a distinct modality. Convention: goal_mask 1 = attend the
        # goal, 0 = undirected (goal tokens replaced by a learned null token).
        if use_goal:
            self.goal_resampler1 = PerceiverResampler(d_model, num_latents=num_goal_latents, nhead=nhead)
            self.goal_resampler2 = PerceiverResampler(d_model, num_latents=num_goal_latents, nhead=nhead)
            self.goal_slot_emb = nn.Parameter(torch.randn(num_goal_latents, d_model) * 0.02)
            self.goal_modality_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)  # room1/room2 goal
            self.goal_null_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Raw-LiDAR-point branch (Option A): a point-set encoder, NOT a BEV CNN.
        # Each (x,y) point -> MLP embedding -> Perceiver resampler (masked over the
        # zero-padded points) -> num_lidar_latents tokens. Preserves mm info that
        # rasterization would quantize away, and matches the online raw scan.
        self.use_lidar_points = use_lidar_points
        self.num_lidar_latents = num_lidar_latents
        if use_lidar_points:
            self.point_proj = nn.Sequential(
                nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.lidar_resampler = PerceiverResampler(d_model, num_latents=num_lidar_latents, nhead=nhead)
            self.lidar_slot_emb = nn.Parameter(torch.randn(num_lidar_latents, d_model) * 0.02)
            self.lidar_modality_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # ICP-distilled aux head: predicts the dock pose [x_norm, y_norm, sin, cos]
        # from the readout vector -> precision/arrival judgment (replaces runtime
        # ICP). Trained on the reliable ICP labels (masked loss in the train loop).
        self.use_aux_pose = use_aux_pose
        if use_aux_pose:
            if use_lidar_points:
                # ★ cross-attention precision head: pose query → raw point tokens
                self.aux_pose_head = CrossAttnPoseHead(d_model, nhead)
            else:
                # fallback (no lidar): pooled-readout MLP head
                self.aux_head = nn.Sequential(
                    nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 4))
            self._aux_pred = None
        self._point_tokens = None        # per-point tokens for the cross-attn head
        self._point_mask = None

        self.readout_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_pos_emb = SinusoidalEmbedding(d_model)

        self.tfm = Transformer(
            d_model,
            nhead,
            num_layers=num_layers,
            attn_dropout=dropout,
            ffn_dropout=dropout,
        )

    def _apply_branch_dropout(self, tokens: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if (not self.training) or drop_prob <= 0.0:
            return tokens

        b = tokens.shape[0]
        keep = (torch.rand(b, device=tokens.device) > drop_prob).float().view(b, 1, 1)
        return tokens * keep

    def _build_image_tokens(
        self,
        dino_feat: torch.Tensor,
        resampler: PerceiverResampler,
        modality_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            dino_feat: [B, Tv, 196, 768]

        Returns:
            [B, Tv * num_image_latents, d_model]
        """
        b, t_seq, n_patch, feat_dim = dino_feat.shape
        if t_seq != self.vision_horizon:
            raise ValueError(
                f"Expected vision_horizon={self.vision_horizon}, but got dino_feat with T={t_seq}."
            )
        if n_patch != 196:
            raise ValueError(f"Expected 196 DINO patches, but got {n_patch}.")
        if feat_dim != 768:
            raise ValueError(f"Expected DINO feature dim=768, but got {feat_dim}.")

        feat_flat = dino_feat.reshape(b * t_seq, n_patch, feat_dim).float()
        patch_tokens = self.vision_proj(feat_flat)
        latent_tokens = resampler(patch_tokens, cond=None)
        latent_tokens = latent_tokens.view(b, t_seq, self.num_image_latents, self.d_model)

        latent_tokens = (
            latent_tokens
            + self.vision_time_emb.view(1, t_seq, 1, self.d_model)
            + self.image_slot_emb.view(1, 1, self.num_image_latents, self.d_model)
            + self.modality_emb[modality_idx].view(1, 1, 1, self.d_model)
        )
        return latent_tokens.reshape(b, t_seq * self.num_image_latents, self.d_model)

    def _build_velocity_tokens(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            velocity: [B, Tm, 2]

        Returns:
            [B, Tm, d_model]
        """
        b, t_seq, dim = velocity.shape
        if t_seq != self.obs_horizon:
            raise ValueError(
                f"Expected obs_horizon={self.obs_horizon}, but got velocity with T={t_seq}."
            )
        if dim != self.velocity_dim:
            raise ValueError(f"Expected velocity_dim={self.velocity_dim}, but got {dim}.")

        vel_tokens = self.velocity_proj(velocity.float())
        vel_tokens = (
            vel_tokens
            + self.motion_time_emb.view(1, t_seq, self.d_model)
            + self.modality_emb[2].view(1, 1, self.d_model)
        )
        vel_tokens = self._apply_branch_dropout(vel_tokens, self.velocity_dropout_prob)
        return vel_tokens

    def _build_goal_tokens(
        self,
        goal_feat1: torch.Tensor,
        goal_feat2: torch.Tensor,
        goal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode the goal DINO features into goal tokens (NoMaD-masked).

        Args:
            goal_feat{1,2}: [B, 1, 196, 768] or [B, 196, 768]  (single goal frame)
            goal_mask: [B] with 1 = attend goal, 0 = undirected (null token)
        Returns:
            [B, 2 * num_goal_latents, d_model]
        """
        def one(feat, resampler, modality_idx):
            if feat.dim() == 4:
                feat = feat[:, 0]                       # drop the singleton time axis
            tok = resampler(self.vision_proj(feat.float()))
            return (
                tok
                + self.goal_slot_emb.view(1, self.num_goal_latents, self.d_model)
                + self.goal_modality_emb[modality_idx].view(1, 1, self.d_model)
            )

        goal_tokens = torch.cat(
            [one(goal_feat1, self.goal_resampler1, 0),
             one(goal_feat2, self.goal_resampler2, 1)], dim=1)

        if goal_mask is not None:
            m = goal_mask.view(-1, 1, 1).float()
            goal_tokens = goal_tokens * m + self.goal_null_emb * (1.0 - m)
        return goal_tokens

    def _build_lidar_tokens(self, points: torch.Tensor, npoints: torch.Tensor) -> torch.Tensor:
        """Encode a padded raw point set into lidar tokens.

        Args:
            points:  [B, M, 2] robot-frame points, zero-padded.
            npoints: [B] number of valid points per sample.
        Returns:
            [B, num_lidar_latents, d_model]
        """
        b, m, _ = points.shape
        tok = self.point_proj(points.float())                         # [B, M, d]
        valid = torch.arange(m, device=points.device)[None, :] < npoints.view(-1, 1)
        pad_mask = ~valid                                             # True = ignore
        pad_mask[npoints <= 0, 0] = False                            # avoid all-masked NaN
        self._point_tokens = tok                                     # for the cross-attn aux head
        self._point_mask = pad_mask
        lat = self.lidar_resampler(tok, key_padding_mask=pad_mask)    # [B, n_lat, d]
        return (lat
                + self.lidar_slot_emb.view(1, self.num_lidar_latents, self.d_model)
                + self.lidar_modality_emb.view(1, 1, self.d_model))

    def forward(self, condition: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        required_keys = ("dino_feat1", "dino_feat2", "velocity")
        missing_keys = [k for k in required_keys if k not in condition]
        if missing_keys:
            raise KeyError(f"Missing condition keys: {missing_keys}")

        dino_feat1 = condition["dino_feat1"]
        dino_feat2 = condition["dino_feat2"]
        velocity = condition["velocity"]

        b = dino_feat1.shape[0]
        device = dino_feat1.device
        self._point_tokens = None        # reset; set by _build_lidar_tokens if lidar present

        image1_tokens = self._build_image_tokens(dino_feat1, self.room1_resampler, modality_idx=0)
        image2_tokens = self._build_image_tokens(dino_feat2, self.room2_resampler, modality_idx=1)
        velocity_tokens = self._build_velocity_tokens(velocity)

        # TODO: turn off pos/vel tokens
        # image1_tokens = torch.zeros_like(image1_tokens)
        # image2_tokens = torch.zeros_like(image2_tokens)
        # velocity_tokens = torch.zeros_like(velocity_tokens)

        readout = self.readout_emb.repeat(b, 1, 1) + self.modality_emb[3].view(1, 1, self.d_model)

        token_list = [image1_tokens, image2_tokens, velocity_tokens]
        if self.use_lidar_points and "lidar_points" in condition:
            token_list.append(self._build_lidar_tokens(
                condition["lidar_points"], condition["lidar_npoints"]))
        if self.use_goal and "goal_feat1" in condition:
            token_list.append(self._build_goal_tokens(
                condition["goal_feat1"], condition["goal_feat2"], condition.get("goal_mask")))
        token_list.append(readout)

        all_tokens = torch.cat(token_list, dim=1)
        token_idx = torch.arange(all_tokens.shape[1], device=device)
        all_tokens = all_tokens + self.token_pos_emb(token_idx).unsqueeze(0)

        fused = self.tfm(all_tokens)[0]
        out = fused[:, -1, :]

        # Cache the aux dock-pose prediction (read by the train loop for the masked
        # precision loss). Cross-attn head when lidar present (direct point access);
        # else pooled-readout MLP fallback.
        if self.use_aux_pose:
            if self.use_lidar_points and self._point_tokens is not None:
                self._aux_pred = self.aux_pose_head(self._point_tokens, self._point_mask, out)
            else:
                self._aux_pred = self.aux_head(out)

        if mask is not None:
            out = out * mask.view(b, 1).float()

        return out
