from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import SinusoidalEmbedding, Transformer


# ==============================================================================
# 1. Perceiver Resampler (기존 유지)
# ==============================================================================
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
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        b = x.shape[0]
        q = self.latents.repeat(b, 1, 1)

        if cond is not None:
            q = q + cond

        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        x = q + attn_out
        x = x + self.ffn(self.ln_post(x))
        return x


class SensorFusionConditionNetwork(BaseNNCondition):
    def __init__(self, state_dim=2, obs_horizon=30, d_model=768, nhead=6, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model, self.obs_horizon, self.dropout = d_model, obs_horizon, dropout

        # 1. Encoder State Tokenizer (for 30 steps)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.LeakyReLU(), nn.Linear(d_model, d_model))
        self.state_temporal_emb = nn.Parameter(torch.randn(obs_horizon, d_model) * 0.02)

        # 2. Vision Tokenizer (for single frame)
        self.vision_proj = nn.Linear(768, d_model)
        self.goal_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LeakyReLU(), nn.Linear(d_model, d_model))

        self.room1_resampler = PerceiverResampler(d_model, 16, nhead)
        self.room2_resampler = PerceiverResampler(d_model, 16, nhead)

        # 3. Fusion & Readout
        self.readout_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.tfm = Transformer(d_model, nhead, num_layers=num_layers, attn_dropout=0.1, ffn_dropout=0.1)
        self.pos_emb = SinusoidalEmbedding(d_model)

    def forward(self, condition: Dict, mask: Optional[torch.Tensor] = None):
        b = condition["state"].shape[0]
        device = condition["state"].device
        tokens = []

        # --- 1. Encoder Sequence (30 Tokens) ---
        state_tokens = self.state_proj(condition["state"].float())  # [b, 30, d_model]
        state_tokens = state_tokens + self.state_temporal_emb.unsqueeze(0)

        if self.training:
            state_mask = torch.bernoulli(torch.full((b, self.obs_horizon, 1), 1 - self.dropout, device=device))
            state_tokens = state_tokens * state_mask

        state_tokens = torch.zeros_like(state_tokens)
        tokens.append(state_tokens)

        # --- 2. Vision (Single Frame -> 16 Tokens each) ---
        def process_vision(dino_feat, sim_map, has_obj, resampler):
            v_feats = self.vision_proj(dino_feat.float())  # [b, 196, d_model]
            s_map = F.interpolate(sim_map.float().unsqueeze(1), size=(14, 14), mode='bilinear').view(b, 196, 1)
            weights = F.softmax(s_map / 0.1, dim=1)
            # 가중 합산된 특징에 goal projection 적용 후 has_obj 마스킹
            goal_cond = self.goal_proj(torch.sum(v_feats * weights, dim=1, keepdim=True)) * has_obj.view(b, 1,
                                                                                                         1).float()
            return resampler(v_feats, cond=goal_cond)

        tokens.append(process_vision(condition["dino_feat1"], condition["sim_map1"], condition["has_object1"],
                                     self.room1_resampler))
        tokens.append(process_vision(condition["dino_feat2"], condition["sim_map2"], condition["has_object2"],
                                     self.room2_resampler))

        # --- 3. Readout & Fusion ---
        tokens.append(self.readout_emb.repeat(b, 1, 1))
        all_tokens = torch.cat(tokens, dim=1)  # [b, 30+16+16+1, d_model]
        all_tokens = all_tokens + self.pos_emb(torch.arange(all_tokens.shape[1], device=device)).unsqueeze(0)

        fused = self.tfm(all_tokens)[0]
        return fused[:, -1, :]
