# Friday Architecture

## Design (research-backed)

- MobileLLM-Flash: shallow-and-wide, skip-attention for latency Pareto optimum.
- GQA (4 KV heads) for KV-cache reduction.
- Shared tied embeddings (~25% param save).
- RoPE, RMSNorm, SwiGLU.
- Full attention only on selected layers; others skip attention (FFN only).

## Mobile budget

weights (INT4) + INT8 KV cache (4k ctx) + ~400MB runtime/OS < 1GB.

## Training stages

Pretrain -> Mid-train -> Reasoning SFT -> Tool SFT -> GRPO -> DPO -> QAT -> Export.
