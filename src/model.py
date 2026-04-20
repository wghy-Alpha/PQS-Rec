from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    num_users: int
    num_items: int
    # 约定 0 为 padding item id
    padding_item_id: int = 0
    # 协同过滤 ID 向量维度
    id_dim: int = 128
    # 物品文本向量维度（PPT 中是 768）
    text_dim: int = 768
    # RQ/KMeans 语义码本参数：768 = 48 * 16
    num_subspaces: int = 48
    subspace_dim: int = 16
    num_centroids: int = 256
    # 序列建模与融合维度
    seq_hidden_dim: int = 128
    fusion_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    # 长尾平滑权重
    w_max: float = 10.0
    eps: float = 1e-8
    # 融合策略：gate / concat_mlp / mean
    fusion_strategy: str = "gate"


def long_tail_weight(popularity: torch.Tensor, eps: float, w_max: float) -> torch.Tensor:
    """
    PPT 公式：w_i = min(log(1 / (p_i + eps)), w_max)
    popularity 即 p_i。
    """
    cap = torch.full_like(popularity, fill_value=w_max)
    return torch.minimum(torch.log(1.0 / (popularity + eps)), cap)


def info_nce_loss(view_a: torch.Tensor, view_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    双塔对比学习（用于语义/协同两视角对齐）。
    """
    a = F.normalize(view_a, dim=-1)
    b = F.normalize(view_b, dim=-1)
    logits = torch.matmul(a, b.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_ab + loss_ba)


class SemanticCodebookQuantizer(nn.Module):
    """
    RQ/KMeans 思路的极致优化版：
    通过高级索引和 cdist 消除临时大张量广播，彻底解决显存带宽瓶颈。
    """

    def __init__(self, num_subspaces: int, subspace_dim: int, num_centroids: int) -> None:
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.num_centroids = num_centroids
        self.codebooks = nn.Parameter(
            torch.randn(num_subspaces, num_centroids, subspace_dim) * 0.02
        )

    def forward(self, text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_emb.size(0)
        split = text_emb.view(batch_size, self.num_subspaces, self.subspace_dim)

        # 【性能优化 1：使用 torch.cdist 替代暴力的 unsqueeze 广播】
        # split_t 形状: [S, B, D], codebooks 形状: [S, C, D]
        split_t = split.transpose(0, 1)
        # 极速计算距离，dist_t 形状: [S, B, C]
        dist_t = torch.cdist(split_t, self.codebooks, p=2.0)
        # 转回 [B, S, C]
        dist = dist_t.transpose(0, 1)
        semantic_ids = dist.argmin(dim=-1)

        # 【性能优化 2：使用 PyTorch 高级索引替代暴力的 expand + gather】
        # s_idx 形状: [S, 1]
        s_idx = torch.arange(self.num_subspaces, device=text_emb.device).unsqueeze(1)
        # 直接通过坐标索引抽取出中心向量，quantized_t 形状: [S, B, D]
        quantized_t = self.codebooks[s_idx, semantic_ids.transpose(0, 1)]

        # 整理回所需形状 [B, S*D]
        quantized = quantized_t.transpose(0, 1).reshape(batch_size, self.num_subspaces * self.subspace_dim)

        return {"quantized": quantized, "semantic_ids": semantic_ids}


class SequenceEncoder(nn.Module):
    """
    SASRec 风格的序列编码骨架（Transformer Encoder）。
    输入 [B, L, D]，输出用户在该视角下的序列表征 [B, D]。
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, seq_emb: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.encoder(seq_emb, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            return out[:, -1, :]
        valid_len = (~padding_mask).sum(dim=1).clamp_min(1)
        last_idx = valid_len - 1
        batch_idx = torch.arange(out.size(0), device=out.device)
        return out[batch_idx, last_idx, :]


class FlexibleFusion(nn.Module):
    """
    两路融合策略：
    - gate: g = sigmoid(W[e_id;e_s]+b), e = g*e_id + (1-g)*e_s
    - concat_mlp: 拼接后过 MLP
    - mean: 简单平均
    """

    def __init__(self, dim_a: int, dim_b: int, out_dim: int, strategy: str = "gate") -> None:
        super().__init__()
        self.strategy = strategy
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate = nn.Linear(out_dim * 2, out_dim)
        self.concat_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)
        if self.strategy == "mean":
            return 0.5 * (a_proj + b_proj)
        if self.strategy == "concat_mlp":
            return self.concat_mlp(torch.cat([a_proj, b_proj], dim=-1))
        g = torch.sigmoid(self.gate(torch.cat([a_proj, b_proj], dim=-1)))
        return g * a_proj + (1.0 - g) * b_proj


class RLMRecFramework(nn.Module):
    """
    对齐 PPT 的整体框架：
    1) 物品 text -> 语义码本量化（semantic ids + semantic embedding）
     2) 用户侧两路：
         - 语义路：历史交互物品语义向量（长尾平滑加权平均）
       - 协同路：历史 item id -> 序列编码
    3) 物品侧两路：语义路 / 协同路
    4) 门控融合 user/item 双路表示，再做打分
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.num_subspaces * cfg.subspace_dim == cfg.text_dim

        # item id embedding（协同过滤路）
        self.item_id_embedding = nn.Embedding(
            cfg.num_items,
            cfg.id_dim,
            padding_idx=cfg.padding_item_id,
        )

        # 语义码本量化模块
        self.quantizer = SemanticCodebookQuantizer(
            num_subspaces=cfg.num_subspaces,
            subspace_dim=cfg.subspace_dim,
            num_centroids=cfg.num_centroids,
        )

        # 协同序列映射到序列建模空间
        self.hist_cf_proj = nn.Linear(cfg.id_dim, cfg.seq_hidden_dim)

        # 用户两路编码（语义为加权平均，协同为序列编码）
        self.user_cf_encoder = SequenceEncoder(cfg.seq_hidden_dim, cfg.num_heads, cfg.num_layers, cfg.dropout)

        # 用户两路头
        self.user_sem_head = nn.Sequential(nn.Linear(cfg.text_dim, cfg.fusion_dim), nn.ReLU())
        self.user_cf_head = nn.Sequential(nn.Linear(cfg.seq_hidden_dim, cfg.fusion_dim), nn.ReLU())
        self.user_profile_head = nn.Sequential(nn.Linear(cfg.text_dim, cfg.fusion_dim), nn.ReLU())
        self.user_sem_profile_fusion = FlexibleFusion(
            cfg.fusion_dim,
            cfg.fusion_dim,
            cfg.fusion_dim,
            cfg.fusion_strategy,
        )

        # 物品两路：先投影到序列空间，再过 SASRec 风格编码器，最后 MLP
        self.item_sem_proj = nn.Linear(cfg.text_dim, cfg.seq_hidden_dim)
        self.item_cf_proj = nn.Linear(cfg.id_dim, cfg.seq_hidden_dim)
        self.item_sem_encoder = SequenceEncoder(cfg.seq_hidden_dim, cfg.num_heads, cfg.num_layers, cfg.dropout)
        self.item_cf_encoder = SequenceEncoder(cfg.seq_hidden_dim, cfg.num_heads, cfg.num_layers, cfg.dropout)
        self.item_sem_head = nn.Sequential(nn.Linear(cfg.seq_hidden_dim, cfg.fusion_dim), nn.ReLU())
        self.item_cf_head = nn.Sequential(nn.Linear(cfg.seq_hidden_dim, cfg.fusion_dim), nn.ReLU())

        # user/item 侧融合
        self.user_fusion = FlexibleFusion(cfg.fusion_dim, cfg.fusion_dim, cfg.fusion_dim, cfg.fusion_strategy)
        self.item_fusion = FlexibleFusion(cfg.fusion_dim, cfg.fusion_dim, cfg.fusion_dim, cfg.fusion_strategy)

    def encode_history_semantic(
        self,
        hist_item_text: torch.Tensor,
        hist_popularity: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        用户语义向量：
        - 每个历史物品文本先做语义码本量化
        - 基于长尾平滑权重做序列加权平均
        """
        batch_size, seq_len, text_dim = hist_item_text.shape
        flat_text = hist_item_text.reshape(batch_size * seq_len, text_dim)
        q = self.quantizer(flat_text)["quantized"].reshape(batch_size, seq_len, text_dim)

        # w_i = min(log(1/(p_i+eps)), w_max)，并在序列维做归一化（忽略 padding）
        weights = long_tail_weight(hist_popularity, self.cfg.eps, self.cfg.w_max)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.cfg.eps)
        user_sem = (q * weights.unsqueeze(-1)).sum(dim=1)
        return user_sem

    def forward(
        self,
        target_item_ids: torch.Tensor,
        target_item_text: torch.Tensor,
        hist_item_ids: torch.Tensor,
        hist_item_text: torch.Tensor,
        hist_popularity: torch.Tensor,
        target_item_popularity: Optional[torch.Tensor] = None,
        user_profile_text: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # padding 位置不参与序列注意力
        padding_mask = hist_item_ids.eq(self.cfg.padding_item_id)

        # ---------- 用户语义路 ----------
        user_sem_vec = self.encode_history_semantic(hist_item_text, hist_popularity, padding_mask)
        user_sem = self.user_sem_head(user_sem_vec)
        if user_profile_text is not None:
            # 将 LLM 生成的用户画像向量注入语义侧，补充历史序列无法覆盖的长期偏好
            user_profile_sem = self.user_profile_head(user_profile_text)
            user_sem = self.user_sem_profile_fusion(user_sem, user_profile_sem)

        # ---------- 用户协同路 ----------
        hist_id_emb = self.item_id_embedding(hist_item_ids)
        hist_cf_tokens = self.hist_cf_proj(hist_id_emb)
        user_cf_seq = self.user_cf_encoder(hist_cf_tokens, padding_mask)
        user_cf = self.user_cf_head(user_cf_seq)

        # ---------- 物品语义路（semantic -> SASRec -> MLP） ----------
        target_q = self.quantizer(target_item_text)["quantized"]
        item_sem_tokens = self.item_sem_proj(target_q).unsqueeze(1)
        item_sem_seq = self.item_sem_encoder(item_sem_tokens, padding_mask=None)
        item_sem = self.item_sem_head(item_sem_seq)

        # ---------- 物品协同路（CF -> SASRec -> MLP） ----------
        target_id_emb = self.item_id_embedding(target_item_ids)
        item_cf_tokens = self.item_cf_proj(target_id_emb).unsqueeze(1)
        item_cf_seq = self.item_cf_encoder(item_cf_tokens, padding_mask=None)
        item_cf = self.item_cf_head(item_cf_seq)

        # ---------- 双路融合 ----------
        user_final = self.user_fusion(user_cf, user_sem)
        item_final = self.item_fusion(item_cf, item_sem)

        logits = (user_final * item_final).sum(dim=-1)

        return {
            "logits": logits,
            "user_sem": user_sem,
            "user_cf": user_cf,
            "item_sem": item_sem,
            "item_cf": item_cf,
        }
