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


# ==============================================================================
# 2. DINOv3 Vision Stem (기존 유지)
# ==============================================================================
class DINOv3VisionStem(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(768, d_model)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        with torch.no_grad():
            outputs = self.backbone(x)
            x = outputs.last_hidden_state[:, 5:, :]
        return self.proj(x)


# ==============================================================================
# 3. Mobile Robot Condition Network (최종 수정본)
# ==============================================================================
class SensorFusionConditionNetwork(BaseNNCondition):
    def __init__(
            self,
            state_dim: int = 9,
            image_size: Tuple[int, int] = (240, 320),
            d_model: int = 768,
            nhead: int = 6,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_vision_tokens: int = 16,
    ):
        super().__init__()
        self.dropout = dropout

        # 1. State 토크나이저
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )
        self.state_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # Goal 특징 프로젝션
        self.goal_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. 비전 토크나이저
        self.vision_room1_stem = DINOv3VisionStem(d_model=d_model)
        self.vision_room1_resampler = PerceiverResampler(d_model, num_vision_tokens, nhead)
        self.vision_room1_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        self.vision_room2_stem = DINOv3VisionStem(d_model=d_model)
        self.vision_room2_resampler = PerceiverResampler(d_model, num_vision_tokens, nhead)
        self.vision_room2_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # 4. 최종 융합 Transformer
        self.readout_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.tfm = Transformer(d_model, nhead, num_layers=4, attn_dropout=0.0, ffn_dropout=0.0)
        self.pos_emb = SinusoidalEmbedding(d_model)

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        b = condition["state"].shape[0]
        device = condition["state"].device
        tokens = []

        # ---------------------------------------------------------
        # 변경 포인트 1: State(Encoder) 토큰에만 선택적 드롭아웃 적용
        # ---------------------------------------------------------
        state_token = self.state_proj(condition["state"].float()).unsqueeze(1)
        state_token = state_token + self.state_emb

        if self.training:
            # 설정된 dropout 확률로 Encoder 정보만 0으로 만듦 (비전은 유지)
            state_mask = torch.bernoulli(torch.full((b, 1, 1), 1 - self.dropout, device=device))
            state_token = state_token * state_mask

        tokens.append(state_token)

        # ---------------------------------------------------------
        # 변경 포인트 2: Room1 전용 가이드(sim_map1) 적용
        # ---------------------------------------------------------
        sim_map1 = condition["sim_map1"].float().unsqueeze(1)
        weights1 = F.interpolate(sim_map1, size=(14, 14), mode='bilinear').view(b, 196, 1)
        weights1 = F.softmax(weights1 / 0.1, dim=1)
        has_obj1 = condition["has_object1"].float().view(b, 1, 1)

        v1_feats = self.vision_room1_stem(condition["image_room1"].float())
        goal_feat_v1 = torch.sum(v1_feats * weights1, dim=1, keepdim=True)
        goal_cond_v1 = self.goal_proj(goal_feat_v1) * has_obj1

        v1_compressed = self.vision_room1_resampler(v1_feats, cond=goal_cond_v1)
        tokens.append(v1_compressed + self.vision_room1_emb)

        # ---------------------------------------------------------
        # 변경 포인트 3: Room2 전용 가이드(sim_map2) 적용
        # ---------------------------------------------------------
        sim_map2 = condition["sim_map2"].float().unsqueeze(1)
        weights2 = F.interpolate(sim_map2, size=(14, 14), mode='bilinear').view(b, 196, 1)
        weights2 = F.softmax(weights2 / 0.1, dim=1)
        has_obj2 = condition["has_object2"].float().view(b, 1, 1)

        v2_feats = self.vision_room2_stem(condition["image_room2"].float())
        goal_feat_v2 = torch.sum(v2_feats * weights2, dim=1, keepdim=True)
        goal_cond_v2 = self.goal_proj(goal_feat_v2) * has_obj2

        v2_compressed = self.vision_room2_resampler(v2_feats, cond=goal_cond_v2)
        tokens.append(v2_compressed + self.vision_room2_emb)

        # 4. Readout 토큰 및 최종 융합
        readout_token = self.readout_emb.repeat(b, 1, 1)
        tokens.append(readout_token)

        all_tokens = torch.cat(tokens, dim=1)
        pos_embedding = self.pos_emb(torch.arange(all_tokens.shape[1], device=device)).unsqueeze(0)

        # 위치 임베딩 추가 (전체 마스킹 cfg_mask_unsqueeze는 제거됨)
        all_tokens = all_tokens + pos_embedding

        mask_cache = torch.tril(torch.ones(all_tokens.shape[1], all_tokens.shape[1], device=device))
        tfm_out = self.tfm(all_tokens, mask_cache)[0]

        return tfm_out[:, -1, :]