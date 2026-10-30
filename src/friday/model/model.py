import torch
import torch.nn as nn
from .config import FridayConfig
from .transformer import FridayBlock
from .norm import RMSNorm
from .rope import compute_rope_params


class FridayModel(nn.Module):
    def __init__(self, cfg: FridayConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([FridayBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = compute_rope_params(cfg.head_dim, cfg.rope_theta, cfg.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = x @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(x)
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
