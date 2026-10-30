from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class FridayConfig:
    vocab_size: int = 128256
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 32
    n_kv_heads: int = 8
    hidden_dim: int = 4096
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    local_window: int = 512
    global_every: int = 4
    full_attn_idxs: Optional[List[int]] = None
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [i for i in range(self.n_layers) if i % self.global_every == 0 or i == self.n_layers - 1]
        self._full_set = set(self.full_attn_idxs)

    def is_full_attn(self, idx: int) -> bool:
        return idx in self._full_set

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def skip_attention(self, idx: int) -> bool:
        return idx not in self._full_set
