# bitsandbytes on RDNA4 (gfx1201)

## Status
bitsandbytes 0.50.0.dev0 supports gfx1201 (RDNA4). Stable release does NOT.

## Build from Source

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
source <your-venv>/bin/activate
export ROCM_PATH=/opt/rocm-7.2.1
export GPU_TARGETS=gfx1201
cmake -DCOMPUTE_BACKEND=hip -DGPU_TARGETS=gfx1201 -S . -B build
cmake --build build -j$(nproc)
pip install -e .
```

## Validation

```python
import bitsandbytes as bnb, torch
x = torch.randn(4, 64).cuda().half()
layer4 = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16).cuda()
out = layer4(x)
print("QLoRA 4-bit: PASSED", out.shape)
layer8 = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False).cuda()
out8 = layer8(x)
print("LLM.int8(): PASSED", out8.shape)
```

## Tested
- ROCm 7.2.1
- RX 9070 (gfx1201, RDNA4)
- Ubuntu 24.04
- Date: 2026-03-27
