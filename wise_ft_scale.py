"""WiSE-FT style interpolation for LoRA adapters (robust fine-tuning).

WiSE-FT (Wortsman et al., CVPR 2022): interpolating between the base (zero-shot)
weights and the fine-tuned weights often recovers OOD robustness at very little
in-distribution cost. For a LoRA adapter this is nearly free:

    merged = base + (lora_alpha / r) * B @ A        # the LoRA contribution
    => scaling every lora_B tensor by s  ==  interpolating dW by s
       (equivalently: merged(s) = base + s * dW)

modules_to_save (e.g. the fully fine-tuned visual merger) are NOT covered by
lora_B scaling — they are interpolated explicitly against the base weights:

    w(s) = s * w_finetuned + (1 - s) * w_base

Output: for each alpha, a drop-in adapter dir compatible with the existing
merge pipeline (merge_qwen35.sh / llamafactory export). No training, CPU-only.

Usage:
    python wise_ft_scale.py \\
        --adapter_dir /path/to/checkpoint-XXXX \\
        --base_model  /path/to/Qwen3_5-9B \\
        --alphas 0.95 0.9 0.85 0.8 0.7 \\
        --out_root    /path/to/wise_ft_out

Then for each <out_root>/alpha_<a>/ run the usual merge + inference + eval.
Pick the alpha with the best Recall@Test2 subject to Test1 P/R >= 0.98.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def find_adapter_file(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if p.exists():
            return p
    raise SystemExit(f"❌ no adapter_model.safetensors/.bin under {adapter_dir}")


def load_adapter_sd(path: Path) -> dict:
    if path.suffix == ".safetensors":
        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def mts_to_base_key(adapter_key: str) -> str:
    """base_model.model.model.visual.merger.modules_to_save.default.linear_fc1.weight
    -> model.visual.merger.linear_fc1.weight (matches base safetensors index)."""
    k = adapter_key
    if k.startswith("base_model.model."):
        k = k[len("base_model.model."):]
    return k.replace(".modules_to_save.default", "")


def load_base_tensors(base_dir: Path, keys: list) -> dict:
    """Load only the needed tensors from the (sharded) base model."""
    idx_path = base_dir / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.load(open(idx_path))["weight_map"]
    else:
        single = base_dir / "model.safetensors"
        if not single.exists():
            raise SystemExit(f"❌ no safetensors index or single file under {base_dir}")
        weight_map = {k: "model.safetensors" for k in keys}

    by_shard: dict = {}
    for k in keys:
        if k not in weight_map:
            raise SystemExit(f"❌ base model is missing tensor: {k}")
        by_shard.setdefault(weight_map[k], []).append(k)

    out = {}
    for shard, ks in by_shard.items():
        with safe_open(str(base_dir / shard), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="LoRA adapter dir (final or checkpoint-XXXX)")
    ap.add_argument("--base_model", required=True,
                    help="Base model dir (needed to interpolate modules_to_save)")
    ap.add_argument("--alphas", type=float, nargs="+", required=True,
                    help="Interpolation coefficients, e.g. 0.95 0.9 0.8 (1.0 = unchanged)")
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    base_dir = Path(args.base_model)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    adapter_file = find_adapter_file(adapter_dir)
    sd = load_adapter_sd(adapter_file)

    lora_b_keys = [k for k in sd if "lora_B" in k or "lora_embedding_B" in k]
    mts_keys = [k for k in sd if ".modules_to_save." in k]
    other = len(sd) - len(lora_b_keys) - len([k for k in sd if "lora_A" in k or "lora_embedding_A" in k]) - len(mts_keys)
    print(f"[adapter] {adapter_file}")
    print(f"          lora_B tensors: {len(lora_b_keys)}  modules_to_save tensors: {len(mts_keys)}"
          f"  (other non-A/B: {other})")

    base_needed = {k: mts_to_base_key(k) for k in mts_keys}
    base_tensors = load_base_tensors(base_dir, sorted(set(base_needed.values()))) if mts_keys else {}
    if mts_keys:
        print(f"[base   ] loaded {len(base_tensors)} tensors for modules_to_save interpolation")

    for alpha in args.alphas:
        out_dir = out_root / f"alpha_{alpha:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        new_sd = {}
        for k, t in sd.items():
            if "lora_B" in k or "lora_embedding_B" in k:
                new_sd[k] = (t.float() * alpha).to(t.dtype)
            elif ".modules_to_save." in k:
                tb = base_tensors[base_needed[k]]
                if tb.shape != t.shape:
                    raise SystemExit(f"❌ shape mismatch for {k}: ft {tuple(t.shape)} vs base {tuple(tb.shape)}")
                new_sd[k] = (alpha * t.float() + (1.0 - alpha) * tb.float()).to(t.dtype)
            else:
                new_sd[k] = t

        save_file(new_sd, str(out_dir / "adapter_model.safetensors"))

        cfg = adapter_dir / "adapter_config.json"
        if cfg.exists():
            shutil.copy2(cfg, out_dir / "adapter_config.json")
        else:
            print(f"⚠️  {cfg} not found — merge will need it; copy manually")

        print(f"  ↳ alpha={alpha:g}  →  {out_dir}")

    print("\nNext: run your merge script on each alpha dir, then inference + eval.")
    print("Pick the alpha with best Recall@Test2 subject to Test1 P/R >= 0.98.")


if __name__ == "__main__":
    main()
