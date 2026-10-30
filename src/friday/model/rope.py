import torch


def compute_rope_params(head_dim, theta_base, max_seq_len, device=None, dtype=None):
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype if dtype is not None else torch.float32)
    sin = freqs.sin().to(dtype if dtype is not None else torch.float32)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (b, h, t, d)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)  # (1,1,t,d/2)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)
