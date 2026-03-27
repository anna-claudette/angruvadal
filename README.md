# Angruvadal

**RAM-Backed MCP Memory Architecture for Consumer LLM Inference**  
*Codename: Angruvadal*

> 900K token context window. 16GB VRAM. 80–120 tok/s.

---

## What This Is

Angruvadal is an architecture and implementation for extending local LLM inference context far beyond VRAM limits using a two-layer approach:

1. **VRAM-Efficient Inference** — RotorQuant 3-bit KV cache compression (3.7× reduction) combined with expert-aware offloading for MoE models. Keeps everything in-tier, avoids PCIe streaming bottlenecks.

2. **MCP RAM Fleet** — A fleet of small fast LLMs (1–3B) running on CPU, backed by 192GB DDR5, exposed as an MCP server. The flagship model calls tools (`context_store`, `context_retrieve`, `expert_prefetch`) to access an intelligent, queryable memory layer. The 192GB isn't overflow — it's a first-class memory system.

Built on and validated for AMD RDNA4 (RX 9070, gfx1201) with ROCm 7.2.1. Also works on Vulkan.

---

## Hardware Reference Configuration

| Component | Spec |
|-----------|------|
| GPU | AMD RX 9070 — 16GB GDDR6 (RDNA4 / gfx1201) |
| CPU | AMD Ryzen 9 9900X — 12C/24T |
| RAM | 192GB DDR5 |
| OS | Ubuntu 24.04, ROCm 7.2.1 |
| Model | GPT-OSS 20B (OpenAI, MXFP4, Apache 2.0) |

---

## Context Window Math

| Configuration | KV Size | Context Tokens | tok/s |
|---|---|---|---|
| Baseline (FP16 KV, 2.5GB headroom) | 48KB/token | ~52K | 134 |
| + RotorQuant 3-bit | 13KB/token | ~192K | ~125 |
| + RotorQuant + RAM tier | 13KB/token (VRAM) + 192GB RAM | ~900K+ | ~120 |
| + MCP semantic fleet | unlimited semantic retrieval | ∞ effective | ~120 |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  GPU — 16GB VRAM                            │
│  GPT-OSS 20B (13GB)                         │
│  RotorQuant KV cache (hot, compressed)      │
│  → 134 tok/s generation                     │
└─────────────────┬───────────────────────────┘
                  │ MCP tool calls
┌─────────────────▼───────────────────────────┐
│  RAM MCP Server — 192GB DDR5                │
│  ├── Router LLM (1-3B, CPU)                 │
│  ├── Retrieval LLM (1-3B, semantic search)  │
│  ├── Compression LLM (summarize → NVMe)     │
│  └── Expert Prefetch (MoE routing ahead)    │
└─────────────────┬───────────────────────────┘
                  │ cold archive
┌─────────────────▼───────────────────────────┐
│  NVMe — 1.8TB cold storage                  │
└─────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Validated ✅
- GPT-OSS 20B running at 134 tok/s on RX 9070 via llama.cpp Vulkan
- ROCm 7.2.1 benchmarks: first published gfx1201 inference data
- bitsandbytes 0.50.0.dev0: QLoRA + LLM.int8() on RDNA4

### Phase 2 — In Progress 🔨
- RotorQuant KV cache integration into llama.cpp
- ~200–300 lines C++ in attention layer
- Triton kernels confirmed working on gfx1201

### Phase 3 — Planned 📋
- MCP RAM server (Python)
- Tools: `context_store`, `context_retrieve`, `expert_prefetch`
- Fleet model integration

### Phase 4 — Future 🎯
- Full integration + benchmarks
- 128K → 900K context validation
- Architecture paper

---

## Related Work & Benchmarks

| Finding | Link |
|---|---|
| First RDNA4 ROCm 7.2.1 inference benchmarks | r/LocalLLaMA, r/ROCm |
| bitsandbytes gfx1201 build guide | See `docs/bitsandbytes-rdna4.md` |
| GPT-OSS 20B RDNA4 results | See `docs/gpt-oss-20b-benchmarks.md` |
| RotorQuant math | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) |

---

## Name

*Angruvadal* — the Ancestor Blade from Larry Correia's Saga of the Forgotten Warrior. Sentient, stores the memories of every previous Bearer, gives their accumulated skill to the current wielder. The architecture mirrors this: every query draws on accumulated knowledge from all prior sessions. The RAM fleet holds the memories. The GPU flagship fights with their strength.

---

## License

Apache 2.0

---

## Status

Early research / implementation in progress. Hardware: GURTHANG II (AMD Ryzen 9 9900X + RX 9070 + 192GB DDR5). Contributions and hardware-diverse benchmarks welcome.
