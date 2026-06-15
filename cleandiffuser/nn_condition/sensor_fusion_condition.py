from typing import Dict, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import SinusoidalEmbedding, Transformer
from vision.raw_patch_encoder import RawPatchEncoder
from vision.lidar_map_encoder import LidarMapEncoder


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

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N_patch, d_model]
            cond: optional conditioning tensor broadcastable to [B, N_latent, d_model]

        Returns:
            [B, N_latent, d_model]
        """
        b = x.shape[0]
        q = self.latents.repeat(b, 1, 1)

        if cond is not None:
            q = q + cond

        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        out = q + attn_out
        out = out + self.ffn(self.ln_post(out))
        return out


class SensorFusionConditionNetwork(BaseNNCondition):
    """
    Two-camera vision + velocity condition encoder.

    vision_backend = "dino" — condition dict:
        - dino_feat1: [B, Tv, 196, 768]
        - dino_feat2: [B, Tv, 196, 768]
        - velocity:   [B, Tm, 2]   = normalized [v, w]

    vision_backend = "raw_cnn" — condition dict (RGB 0~1, same as dataset):
        - raw_image1: [B, Tv, 3, H, W]
        - raw_image2: [B, Tv, 3, H, W]
        - velocity:   [B, Tm, 2]

    Optional lidar branch (use_lidar=True):
        - lidar_map: [B, Tv, C, S, S]   BEV occupancy map in [0, 1] (same Tv as vision)

    Optional ViNT/NoMaD-style goal branch (use_goal=True), with EARLY goal fusion:
        - raw_cnn backend: goal_image1 / goal_image2: [B, 3, H, W]   (single future frame, [0, 1]).
          Each is stacked channel-wise with the current observation frame
          (raw_image{i}[:, -1]) and encoded by a dedicated 6-channel goal-fusion
          encoder phi(o_t, o_g) -> relative goal token.
        - dino   backend: goal_feat1  / goal_feat2 : [B, 196, 768]   (independent, late-fusion fallback)
        - goal_mask (optional): [B] float in {0, 1}. 1 = attend to goal
          (goal-conditioned), 0 = block goal token (undirected). If absent, a
          mask is sampled internally during training (Bernoulli(goal_mask_prob)
          of being *blocked*) and defaults to 1 (use goal) at eval.
          NOTE: this convention is the inverse of NoMaD's ``m`` (where m=1 masks
          out); here goal_mask=1 keeps the goal active.

    where
        - Tv = vision_horizon (default: 5)
        - Tm = obs_horizon    (default: 30)

    Output:
        - [B, d_model] condition vector from the readout token.
    """

    # Fixed modality embedding slots (kept stable so token tagging never drifts).
    MOD_IMG1 = 0
    MOD_IMG2 = 1
    MOD_VEL = 2
    MOD_READOUT = 3
    MOD_LIDAR = 4
    MOD_GOAL = 5
    N_MODALITIES = 6

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
        vision_backend: str = "raw_cnn",
        use_lidar: bool = True,
        lidar_in_ch: int = 2,
        num_lidar_latents: int = 16,
        lidar_dropout_prob: float = 0.0,
        use_goal: bool = True,
        num_goal_latents: int = 16,
        goal_mask_prob: float = 0.5,
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
        if vision_backend not in ("dino", "raw_cnn"):
            raise ValueError(f"vision_backend must be 'dino' or 'raw_cnn', got {vision_backend!r}")
        self.vision_backend = vision_backend
        self.use_lidar = use_lidar
        self.lidar_in_ch = lidar_in_ch
        self.num_lidar_latents = num_lidar_latents
        self.lidar_dropout_prob = lidar_dropout_prob
        self.use_goal = use_goal
        self.num_goal_latents = num_goal_latents
        self.goal_mask_prob = goal_mask_prob

        if self.vision_backend == "raw_cnn":
            # Shared encoder for both camera streams.
            self.raw_patch_encoder = RawPatchEncoder(in_ch=3, out_dim=768, spatial=14)
        else:
            self.raw_patch_encoder = None

        # Vision branches: one branch per room.
        self.vision_proj = nn.Linear(768, d_model)
        self.room1_resampler = PerceiverResampler(d_model, num_latents=num_image_latents, nhead=nhead)
        self.room2_resampler = PerceiverResampler(d_model, num_latents=num_image_latents, nhead=nhead)

        # Lidar branch (BEV occupancy map -> patch tokens -> resampler).
        if self.use_lidar:
            self.lidar_patch_encoder = LidarMapEncoder(in_ch=lidar_in_ch, out_dim=768, spatial=14)
            self.lidar_proj = nn.Linear(768, d_model)
            self.lidar_resampler = PerceiverResampler(
                d_model, num_latents=num_lidar_latents, nhead=nhead
            )
            self.lidar_time_emb = nn.Parameter(torch.randn(vision_horizon, d_model) * 0.02)
            self.lidar_slot_emb = nn.Parameter(torch.randn(num_lidar_latents, d_model) * 0.02)
        else:
            self.lidar_patch_encoder = None
            self.lidar_proj = None
            self.lidar_resampler = None
            self.lidar_time_emb = None
            self.lidar_slot_emb = None

        # Goal branch (ViNT/NoMaD-style with EARLY goal fusion).
        # ViNT's key insight: goal features should be *relative* to the current
        # observation, so the goal frame is stacked channel-wise with the current
        # observation frame and passed through a dedicated goal-fusion encoder
        # phi(o_t, o_g) -> goal token (NOT encoded independently / "late fusion").
        #   - raw_cnn backend: a separate 6-channel patch encoder ingests
        #     concat([current_frame, goal_frame], dim=channel).
        #   - dino backend: frozen 3-channel ViT cannot ingest a 6-channel stack,
        #     so it falls back to independent goal features (kept for completeness;
        #     this path is not ViNT early-fusion).
        if self.use_goal:
            if self.vision_backend == "raw_cnn":
                self.goal_fusion_encoder = RawPatchEncoder(in_ch=6, out_dim=768, spatial=14)
            else:
                self.goal_fusion_encoder = None
            self.goal_proj = nn.Linear(768, d_model)
            self.goal_resampler = PerceiverResampler(
                d_model, num_latents=num_goal_latents, nhead=nhead
            )
            self.goal_slot_emb = nn.Parameter(torch.randn(num_goal_latents, d_model) * 0.02)
        else:
            self.goal_fusion_encoder = None
            self.goal_proj = None
            self.goal_resampler = None
            self.goal_slot_emb = None

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

        # Fixed-size modality table indexed by the MOD_* constants above
        # (0=image1, 1=image2, 2=velocity, 3=readout, 4=lidar, 5=goal). A couple
        # of unused rows when lidar/goal are off are harmless.
        self.modality_emb = nn.Parameter(torch.randn(self.N_MODALITIES, d_model) * 0.02)

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
            patch_feat: [B, Tv, 196, 768]

        Returns:
            [B, Tv * num_image_latents, d_model]
        """
        b, t_seq, n_patch, feat_dim = dino_feat.shape
        if t_seq != self.vision_horizon:
            raise ValueError(
                f"Expected vision_horizon={self.vision_horizon}, but got patch features with T={t_seq}."
            )
        if n_patch != 196:
            raise ValueError(f"Expected 196 spatial patches, but got {n_patch}.")
        if feat_dim != 768:
            raise ValueError(f"Expected vision feature dim=768, but got {feat_dim}.")

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
            + self.modality_emb[self.MOD_VEL].view(1, 1, self.d_model)
        )
        vel_tokens = self._apply_branch_dropout(vel_tokens, self.velocity_dropout_prob)
        return vel_tokens

    def _encode_raw_to_patches(
        self, raw: torch.Tensor, encoder: RawPatchEncoder
    ) -> torch.Tensor:
        """raw: [B, Tv, 3, H, W] -> [B, Tv, 196, 768]"""
        b, t, c, h, w = raw.shape
        flat = raw.reshape(b * t, c, h, w)
        patches = encoder(flat)
        return patches.view(b, t, 196, 768)

    def _encode_lidar_to_patches(
        self, lidar: torch.Tensor
    ) -> torch.Tensor:
        """lidar: [B, Tv, C, S, S] -> [B, Tv, 196, 768]"""
        b, t, c, h, w = lidar.shape
        flat = lidar.reshape(b * t, c, h, w)
        patches = self.lidar_patch_encoder(flat)
        return patches.view(b, t, 196, 768)

    def _build_lidar_tokens(self, lidar_feat: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """
        Args:
            lidar_feat: [B, Tv, 196, 768]

        Returns:
            [B, Tv * num_lidar_latents, d_model]
        """
        b, t_seq, n_patch, feat_dim = lidar_feat.shape
        if t_seq != self.vision_horizon:
            raise ValueError(
                f"Expected vision_horizon={self.vision_horizon}, but got lidar features with T={t_seq}."
            )
        if n_patch != 196 or feat_dim != 768:
            raise ValueError(
                f"Expected lidar patch shape (*, 196, 768), got (*, {n_patch}, {feat_dim})."
            )

        feat_flat = lidar_feat.reshape(b * t_seq, n_patch, feat_dim).float()
        patch_tokens = self.lidar_proj(feat_flat)
        latent_tokens = self.lidar_resampler(patch_tokens, cond=None)
        latent_tokens = latent_tokens.view(b, t_seq, self.num_lidar_latents, self.d_model)

        latent_tokens = (
            latent_tokens
            + self.lidar_time_emb.view(1, t_seq, 1, self.d_model)
            + self.lidar_slot_emb.view(1, 1, self.num_lidar_latents, self.d_model)
            + self.modality_emb[modality_idx].view(1, 1, 1, self.d_model)
        )
        tokens = latent_tokens.reshape(b, t_seq * self.num_lidar_latents, self.d_model)
        tokens = self._apply_branch_dropout(tokens, self.lidar_dropout_prob)
        return tokens

    def _build_goal_tokens(self, goal_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goal_feat: [B, 196, 768]   (relative goal-fusion features phi(o_t, o_g))

        Returns:
            [B, num_goal_latents, d_model]
        """
        b, n_patch, feat_dim = goal_feat.shape
        if n_patch != 196 or feat_dim != 768:
            raise ValueError(
                f"Expected goal patch shape (*, 196, 768), got (*, {n_patch}, {feat_dim})."
            )

        patch_tokens = self.goal_proj(goal_feat.float())
        latent_tokens = self.goal_resampler(patch_tokens, cond=None)  # [B, num_goal_latents, d_model]
        latent_tokens = (
            latent_tokens
            + self.goal_slot_emb.view(1, self.num_goal_latents, self.d_model)
            + self.modality_emb[self.MOD_GOAL].view(1, 1, self.d_model)
        )
        return latent_tokens

    def _encode_goal_fused(self, condition: Dict, idx: int) -> torch.Tensor:
        """ViNT early goal fusion for room ``idx`` in {1, 2} -> [B, 196, 768].

        Stacks the *current* observation frame (o_t = last vision frame) with the
        goal frame (o_g) channel-wise and runs the dedicated 6-channel goal-fusion
        encoder phi(o_t, o_g), yielding relative goal features. The dino backend
        cannot channel-concat into a frozen ViT, so it falls back to independent
        goal features (``goal_feat{idx}``)."""
        if self.vision_backend == "raw_cnn":
            cur = condition[f"raw_image{idx}"]      # [B, Tv, 3, H, W]
            cur = cur[:, -1]                        # current frame o_t: [B, 3, H, W]
            goal = condition[f"goal_image{idx}"]    # [B, 3, H, W] (or [B, 1, 3, H, W])
            if goal.dim() == 5:
                goal = goal[:, -1]
            fused = torch.cat([cur, goal], dim=1)   # [B, 6, H, W]
            return self.goal_fusion_encoder(fused)  # [B, 196, 768]
        return condition[f"goal_feat{idx}"]

    def forward(self, condition: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.vision_backend == "raw_cnn":
            required_keys = ("raw_image1", "raw_image2", "velocity")
        else:
            required_keys = ("dino_feat1", "dino_feat2", "velocity")
        if self.use_lidar:
            required_keys = required_keys + ("lidar_map",)

        missing_keys = [k for k in required_keys if k not in condition]
        if missing_keys:
            raise KeyError(f"Missing condition keys: {missing_keys}")

        velocity = condition["velocity"]

        if self.vision_backend == "raw_cnn":
            assert self.raw_patch_encoder is not None
            dino_feat1 = self._encode_raw_to_patches(condition["raw_image1"], self.raw_patch_encoder)
            dino_feat2 = self._encode_raw_to_patches(condition["raw_image2"], self.raw_patch_encoder)
        else:
            dino_feat1 = condition["dino_feat1"]
            dino_feat2 = condition["dino_feat2"]

        b = dino_feat1.shape[0]
        device = dino_feat1.device

        image1_tokens = self._build_image_tokens(dino_feat1, self.room1_resampler, modality_idx=self.MOD_IMG1)
        image2_tokens = self._build_image_tokens(dino_feat2, self.room2_resampler, modality_idx=self.MOD_IMG2)
        velocity_tokens = self._build_velocity_tokens(velocity)

        # TODO: turn off pos/vel tokens
        # image1_tokens = torch.zeros_like(image1_tokens)
        # image2_tokens = torch.zeros_like(image2_tokens)
        # velocity_tokens = torch.zeros_like(velocity_tokens)

        readout = self.readout_emb.repeat(b, 1, 1) + self.modality_emb[self.MOD_READOUT].view(1, 1, self.d_model)

        token_groups = [image1_tokens, image2_tokens, velocity_tokens]

        if self.use_lidar:
            lidar_feat = self._encode_lidar_to_patches(condition["lidar_map"])
            lidar_tokens = self._build_lidar_tokens(lidar_feat, modality_idx=self.MOD_LIDAR)
            token_groups.append(lidar_tokens)

        # ------------------------------------------------------------------
        # NoMaD-style goal token + attention masking.
        #   - Goal tokens are appended right before the readout token.
        #   - ``goal_active`` (per-sample, 1=attend / 0=block) is taken from
        #     condition["goal_mask"] if present, else sampled during training
        #     (blocked with prob goal_mask_prob), else defaults to 1 at eval.
        #   - Masking is done at the attention level: every query is blocked
        #     from using the goal tokens as *keys* (so the readout never sees a
        #     masked goal), while observation tokens stay fully visible.
        # ------------------------------------------------------------------
        has_goal_input = self.use_goal and (
            ("goal_image1" in condition and "goal_image2" in condition)
            if self.vision_backend == "raw_cnn"
            else ("goal_feat1" in condition and "goal_feat2" in condition)
        )

        goal_slice = None
        if has_goal_input:
            goal_feat1 = self._encode_goal_fused(condition, 1)
            goal_feat2 = self._encode_goal_fused(condition, 2)
            goal_tokens = torch.cat(
                [self._build_goal_tokens(goal_feat1), self._build_goal_tokens(goal_feat2)], dim=1
            )  # [B, 2*num_goal_latents, d_model]
            g_start = sum(t.shape[1] for t in token_groups)
            g_end = g_start + goal_tokens.shape[1]
            goal_slice = (g_start, g_end)
            token_groups.append(goal_tokens)

        token_groups.append(readout)

        all_tokens = torch.cat(token_groups, dim=1)
        token_idx = torch.arange(all_tokens.shape[1], device=device)
        all_tokens = all_tokens + self.token_pos_emb(token_idx).unsqueeze(0)

        attn_mask = None
        if goal_slice is not None:
            goal_active = self._resolve_goal_active(condition, b, device)  # [B] in {0, 1}
            if not bool(torch.all(goal_active > 0.5)):
                g_start, g_end = goal_slice
                L = all_tokens.shape[1]
                attn_mask = torch.ones(b, L, L, device=device)
                block = goal_active < 0.5  # samples whose goal must be hidden
                # Block goal columns (keys) for every query of the blocked samples.
                attn_mask[block, :, g_start:g_end] = 0.0

        fused = self.tfm(all_tokens, mask=attn_mask)[0]
        out = fused[:, -1, :]

        if mask is not None:
            out = out * mask.view(b, 1).float()

        return out

    def _resolve_goal_active(self, condition: Dict, b: int, device: torch.device) -> torch.Tensor:
        """Per-sample goal activation in {0, 1} (1=attend to goal, 0=block).

        Priority: explicit ``condition["goal_mask"]`` > training-time Bernoulli
        sampling (blocked with prob ``goal_mask_prob``) > default 1 (use goal)."""
        goal_mask = condition.get("goal_mask", None)
        if goal_mask is not None:
            return goal_mask.to(device=device, dtype=torch.float32).view(b)
        if self.training:
            return (torch.rand(b, device=device) > self.goal_mask_prob).float()
        return torch.ones(b, device=device)
