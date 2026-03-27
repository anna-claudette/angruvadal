# GPT-OSS 20B on RDNA4 — Benchmarks

## Hardware
- GPU: AMD RX 9070 16GB (gfx1201, RDNA4)
- CPU: Ryzen 9 9900X
- RAM: 192GB DDR5
- Backend: llama.cpp Vulkan
- ROCm: 7.2.1

## Model
- openai/gpt-oss-20b (Apache 2.0)
- 20.91B params, MoE (32 experts, 4 active)
- MXFP4 quantized weights (baked in)
- GGUF: 12.83 GiB (F16 conversion from safetensors)
- Context window: 131,072 tokens (YaRN RoPE scaling)

## Benchmark Results (llama.cpp Vulkan, flash attention on)

| Test | pp512 (t/s) | tg128 (t/s) |
|------|-------------|-------------|
| Full GPU (-ngl 999, FA on) | 3,574 | **134** |
| Expert offload (-ot exps=CPU) | 625 | 29.7 |

**Recommendation:** Keep all layers on GPU. Expert offload hurts — model fits in 16GB.

## VRAM Budget
- Model: 12.5GB
- KV cache headroom: ~2.5GB → ~52K tokens at FP16
- With RotorQuant 3-bit: ~9.2GB → ~192K tokens (planned)

## Notes
- GPT-OSS 20B uses alternating sliding window (128-token) + full attention layers
- Sliding window KV cost is negligible (6MB for all layers)
- Full attention KV is the bottleneck
- Model runs at 134 tok/s — excellent for a 21B reasoning model on consumer hardware

## Triton Kernel Fix (RDNA4)
GPT-OSS 20B's triton kernels didn't handle gfx1201. Patch:
`target_info.py` in the HF kernel cache — add gfx1201 to `get_cdna_version()` returning 3.
Patch saved at: `~/src/gpt-oss-kernels-patch/restore-patch.sh`
