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
        vision_backend: str = "raw_cnn",
        use_lidar: bool = True,
        lidar_in_ch: int = 2,
        num_lidar_latents: int = 16,
        lidar_dropout_prob: float = 0.0,
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

        # modality indices: 0=image1, 1=image2, 2=velocity, 3=readout, 4=lidar (only if use_lidar)
        n_modalities = 5 if self.use_lidar else 4
        self.modality_emb = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)

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
            + self.modality_emb[2].view(1, 1, self.d_model)
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

        image1_tokens = self._build_image_tokens(dino_feat1, self.room1_resampler, modality_idx=0)
        image2_tokens = self._build_image_tokens(dino_feat2, self.room2_resampler, modality_idx=1)
        velocity_tokens = self._build_velocity_tokens(velocity)

        # TODO: turn off pos/vel tokens
        # image1_tokens = torch.zeros_like(image1_tokens)
        # image2_tokens = torch.zeros_like(image2_tokens)
        # velocity_tokens = torch.zeros_like(velocity_tokens)

        readout = self.readout_emb.repeat(b, 1, 1) + self.modality_emb[3].view(1, 1, self.d_model)

        token_groups = [image1_tokens, image2_tokens, velocity_tokens]

        if self.use_lidar:
            lidar_feat = self._encode_lidar_to_patches(condition["lidar_map"])
            lidar_tokens = self._build_lidar_tokens(lidar_feat, modality_idx=4)
            token_groups.append(lidar_tokens)

        token_groups.append(readout)

        all_tokens = torch.cat(token_groups, dim=1)
        token_idx = torch.arange(all_tokens.shape[1], device=device)
        all_tokens = all_tokens + self.token_pos_emb(token_idx).unsqueeze(0)

        fused = self.tfm(all_tokens)[0]
        out = fused[:, -1, :]

        if mask is not None:
            out = out * mask.view(b, 1).float()

        return out
