# Friday

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabareesh/friday/blob/main/notebooks/train_friday.ipynb)

Fast mobile reasoning model with tool calling, built from scratch by **sabareesh**.

- Name: **Friday**
- Developer: sabareesh
- Target: run smoothly within 1 GB RAM on Android
- Capabilities: research, math, logic reasoning, fast inference, tool calling (MCP read/write), Jarvis-style assistant

## Tiers

| Tier | Params | RAM (INT4) | Use |
|------|--------|------------|-----|
| Friday-Nano | ~140M | ~120 MB | fastest, edge |
| Friday-Flash | ~360M | ~180 MB | default mobile |
| Friday-Base | ~600M | ~300 MB | balanced |

## Architecture

MobileLLM-Flash style: shallow-and-wide transformer, skip-attention layers, GQA, shared embeddings, RoPE, SwiGLU, RMSNorm. See docs/ARCHITECTURE.md.

## Training

1. Pretrain (LM + distillation)
2. Mid-train (domain shift)
3. Reasoning SFT (DeepSeek-R1 distillation)
4. Tool-calling SFT (xLAM, function masking, abstention)
5. GRPO (verifiable rewards)
6. DPO (persona + safety)
7. QAT to INT4
8. Export (GGUF / ExecuTorch)

## MCP tool calling

Friday emits strict single JSON tool calls; host routes to MCP servers (read + write with confirmation gating).

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
sh scripts/train_nano.sh
```

## License

Apache-2.0. Model weights by sabareesh.
