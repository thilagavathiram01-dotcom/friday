import torch
import torch.nn as nn
from .config import FridayConfig
from .attention import GroupedQueryAttention
from .ffn import SwiGLUFeedForward
from .norm import RMSNorm


class FridayBlock(nn.Module):
    def __init__(self, cfg: FridayConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.has_attn = cfg.is_full_attn(layer_idx)
        if self.has_attn:
            self.att = GroupedQueryAttention(cfg.dim, cfg.n_heads, cfg.n_kv_heads)
            self.norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLUFeedForward(cfg.dim, cfg.hidden_dim)
        self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, cos, sin):
        if self.has_attn:
            x = x + self.att(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x
