"""Inspect the f2llm-4b LoRA adapter contents."""
from safetensors import safe_open

path = r"D:/Project/smalljev/evals/arms/f2llm-4b-lora/adapter_model.safetensors"
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print(f"LoRA keys: {len(keys)}")
    total = 0
    for k in keys[:3]:
        t = f.get_tensor(k)
        total += t.numel()
        print(f"  {k}: shape={list(t.shape)} dtype={t.dtype} mean={t.float().mean().item():.6f} std={t.float().std().item():.6f}")
    # Compute totals
    zeros = 0
    for k in keys:
        t = f.get_tensor(k)
        total += t.numel()
        if t.float().abs().max().item() == 0.0:
            zeros += 1
    print("---")
    print(f"Total params in LoRA: {total:,}")
    print(f"All-zero tensors: {zeros}/{len(keys)}")