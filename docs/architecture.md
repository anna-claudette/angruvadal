# Angruvadal: A Heterogeneous Memory Architecture for Consumer LLM Inference

**Codename:** Angruvadal  
**Hardware:** AMD RX 9070 (16GB VRAM) + Ryzen 9 9900X + 192GB DDR5  
**Date:** 2026-03-27  
**Status:** Spec v1 — Ready for implementation

---

## The Core Insight

The 192GB of DDR5 is not slow storage. It is the blade.

Every approach that treats system RAM as a fallback misses the architecture. The correct framing:

> The GPU is a 16GB execution unit. The RAM is a 192GB intelligent memory system. The MCP protocol is the nervous system. Small fleet models living in RAM are the accumulated skill of every previous Bearer. The flagship on the GPU draws the blade — and fights with the strength of all who came before.

---

## Two-Layer Architecture

### Layer 1: GPU Execution (VRAM-Efficient Inference)
- GPT-OSS 20B attention layers resident in VRAM
- Expert weights paged to RAM via `-ot "exps=CPU"` (reclaims ~10GB VRAM)
- RotorQuant 3-bit KV compression (3.7× reduction)
- **Result: ~900K token context, 80-120 tok/s**

### Layer 2: MCP RAM Fleet (Semantic Context)
- Router LLM (1-3B on CPU) — classifies and routes queries
- Retrieval LLM (1-3B on CPU) — semantic search over 192GB context store
- Compression LLM — summarizes old context before NVMe archival
- Expert Predictor — prefetches likely next experts before GPU needs them
- **Result: Effectively unlimited context via semantic retrieval**

---

## Build Sequence

### Phase 1 (Now — zero new code)
- Test `-ot "exps=CPU"` on GPT-OSS 20B GGUF
- Measure VRAM reclaim and tok/s impact
- Baseline: current 134 tok/s with 2.5GB KV headroom

### Phase 2 (This week)
- RotorQuant → llama.cpp C++ patch (~200-300 lines)
- Hook into attention layer: compress K,V after compute, decompress before dot product
- Triton kernels confirmed working on RDNA4 (gfx1201)
- **Publish: first RotorQuant integration on consumer RDNA4**

### Phase 3 (Next sprint)
- MCP RAM server prototype (Python)
- Tools: `context_store`, `context_retrieve`, `expert_prefetch`, `kv_page_write`
- Fleet model integration (Qwen3-1.7B or similar for router/retrieval)

### Phase 4 (Full Angruvadal v1)
- Integrate all layers
- Benchmark 128K → 900K context
- Publish architecture paper

---

## Avenue Analysis Summary

| Avenue | Complexity | Context | tok/s | Viable? |
|--------|-----------|---------|-------|---------|
| Expert offload (`-ot "exps=CPU"`) | Easy | 250K uncompressed | 80-100 | ✅ First |
| RotorQuant KV compression | Medium | 192K VRAM / 900K combined | ~120 | ✅ Second |
| Tiered KV (RAM as L2) | Hard | Unlimited | 10-40 on miss | ⚠️ Only with prefetch |
| HMM unified memory | Easy proto / Hard perf | Same as PCIe | Worse | ⚠️ Prototyping only |
| Flash attention tiling | Medium | Unlimited | 10-40 (PCIe bound) | ⚠️ Memory, not throughput |
| MCP semantic fleet | Medium | Infinite | Negligible overhead | ✅ Layer 2 |

---

## Hardware Reality

| Resource | Bandwidth | Role |
|----------|-----------|------|
| GPU VRAM (16GB) | ~500 GB/s | Model weights + hot KV + active experts |
| System RAM (192GB) | ~80-90 GB/s | Context store + fleet models + cold experts |
| PCIe 4.0 x16 | ~32 GB/s | Bridge (avoid streaming large tensors through here) |
| CPU (9900X) | ~100 GFLOPS AVX2 | Fleet model host |
| NVMe (1.8TB) | ~7 GB/s | Cold archive |

**Key constraint:** PCIe bandwidth (32GB/s) limits raw KV streaming. Architectures that avoid bulk PCIe transfer win. RotorQuant + expert offload keep everything in-tier.

---

## Name

Angruvadal — the Ancestor Blade from Larry Correia's Saga of the Forgotten Warrior. Wields the memories of every previous Bearer. Sentient, chooses its wielder, gives accumulated skill.

The architecture mirrors this exactly: every query draws on accumulated knowledge from all prior sessions. The RAM fleet holds the memories. The GPU flagship fights with their strength.

---

*Forge Kingdom — GURTHANG II — 2026-03-27*
