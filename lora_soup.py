"""Model soup / SWA for LoRA checkpoints (uniform or weighted averaging).

Model soups (Wortsman et al., ICML 2022) and SWA (Izmailov et al., UAI 2018):
averaging weights of multiple fine-tuned snapshots lands in a flatter minimum
and typically improves OOD generalization at zero inference cost.

Two modes:

  --mode adapter   Average raw lora_A / lora_B / modules_to_save tensors across
                   adapters. Output = a normal adapter dir (works with the
                   existing merge pipeline, and composes with wise_ft_scale.py).
                   ✅ Safe for checkpoints OF THE SAME RUN (same lora_A init).
                   ⚠️ Across runs, A-matrices may live in different subspaces;
                   run --check first (reports cross-adapter lora_A cosine sim).
                   Same-seed LlamaFactory runs usually share the init → OK.

  --mode merged    Soup in FULL-WEIGHT space:  W = W_base + Σ wᵢ·ΔWᵢ, where
                   ΔWᵢ = (alphaᵢ/rᵢ)·Bᵢ@Aᵢ;  modules_to_save → Σ wᵢ·W_ft,ᵢ.
                   Streams the base model shard-by-shard; output = a ready
                   merged model dir (no separate merge step needed).
                   ✅ Mathematically sound for ANY mix of runs.

Usage:
    # checkpoint soup within one run (adapter space)
    python lora_soup.py --mode adapter \\
        --adapters ckpt-2000 ckpt-2025 ckpt-2050 \\
        --out_dir soup_adapter_out

    # cross-run soup (full-weight space, e.g. Exp3 + Exp6 + Exp7 finals)
    python lora_soup.py --mode merged \\
        --adapters exp3_adapter exp6_adapter exp7_adapter \\
        --weights 0.2 0.4 0.4 \\
        --base_model /path/to/Qwen3_5-9B \\
        --out_dir soup_merged_model

    # sanity check before cross-run adapter-space soup
    python lora_soup.py --check --adapters exp6_adapter exp7_adapter
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


# ---------------- shared helpers ----------------
def find_adapter_file(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if p.exists():
            return p
    raise SystemExit(f"❌ no adapter_model file under {adapter_dir}")


def load_adapter(adapter_dir: Path):
    f = find_adapter_file(adapter_dir)
    sd = load_file(str(f)) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu")
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    return sd, cfg


def lora_scaling(cfg: dict) -> float:
    r = cfg.get("r", 8)
    alpha = cfg.get("lora_alpha", r)
    if cfg.get("use_rslora", False):
        return alpha / (r ** 0.5)
    return alpha / r


def normalize_weights(weights, n):
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise SystemExit(f"❌ --weights count ({len(weights)}) != adapters count ({n})")
    s = sum(weights)
    return [w / s for w in weights]


# ---------------- --check: cross-adapter lora_A similarity ----------------
def run_check(adapter_dirs):
    sds = [load_adapter(Path(d))[0] for d in adapter_dirs]
    common_a = sorted(set.intersection(*[
        {k for k in sd if "lora_A" in k} for sd in sds
    ]))
    if not common_a:
        print("❌ no common lora_A keys"); return
    sample = common_a[:: max(1, len(common_a) // 8)][:8]
    print(f"[check] {len(common_a)} common lora_A tensors; sampling {len(sample)}")
    print(f"{'tensor':<70} | pairwise cos(A)")
    sims_all = []
    for k in sample:
        vecs = [sd[k].float().flatten() for sd in sds]
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(torch.nn.functional.cosine_similarity(vecs[i], vecs[j], dim=0).item())
        m = sum(sims) / len(sims)
        sims_all.append(m)
        print(f"{k[-70:]:<70} | {m:+.3f}")
    overall = sum(sims_all) / len(sims_all)
    print(f"\n  mean cos(lora_A) = {overall:+.3f}")
    if overall > 0.5:
        print("  ✅ A-subspaces aligned — adapter-space soup is safe.")
    else:
        print("  ⚠️ A-subspaces diverged — prefer --mode merged for these adapters.")


# ---------------- mode: adapter ----------------
def soup_adapter(adapter_dirs, weights, out_dir: Path):
    sds, cfgs = [], []
    for d in adapter_dirs:
        sd, cfg = load_adapter(Path(d))
        sds.append(sd)
        cfgs.append(cfg)

    keysets = [set(sd.keys()) for sd in sds]
    if len(set(map(frozenset, keysets))) != 1:
        raise SystemExit("❌ adapters have different key sets — cannot soup in adapter space")
    for field in ("r", "lora_alpha", "use_rslora"):
        vals = {json.dumps(c.get(field)) for c in cfgs}
        if len(vals) > 1:
            raise SystemExit(f"❌ adapter_config mismatch on '{field}': {vals} — "
                             f"adapter-space soup requires identical configs (use --mode merged)")

    w = normalize_weights(weights, len(sds))
    print(f"[soup/adapter] {len(sds)} adapters, weights={[round(x,4) for x in w]}")

    out = {}
    for k in sds[0]:
        acc = sum(wi * sd[k].float() for wi, sd in zip(w, sds))
        out[k] = acc.to(sds[0][k].dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    shutil.copy2(Path(adapter_dirs[0]) / "adapter_config.json", out_dir / "adapter_config.json")
    print(f"  ↳ {out_dir}  (adapter dir; feed to your usual merge script)")


# ---------------- mode: merged ----------------
def collect_deltas(adapter_dirs, weights):
    """Return (lora_map, mts_map):
       lora_map: base_key -> list of (w_i, scaled ΔW tensor builder inputs)
       mts_map:  base_key -> accumulated Σ w_i * W_ft (float32)"""
    w = normalize_weights(weights, len(adapter_dirs))
    lora_map: dict = {}
    mts_map: dict = {}
    for wi, d in zip(w, adapter_dirs):
        sd, cfg = load_adapter(Path(d))
        s = lora_scaling(cfg)
        for k in sd:
            if k.endswith("lora_A.weight"):
                kb = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
                base_key = kb.replace(".lora_A.weight", ".weight")
                b_key = k.replace("lora_A", "lora_B")
                if b_key not in sd:
                    raise SystemExit(f"❌ missing lora_B for {k}")
                lora_map.setdefault(base_key, []).append((wi, s, sd[k].float(), sd[b_key].float()))
            elif ".modules_to_save." in k:
                kb = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
                base_key = kb.replace(".modules_to_save.default", "")
                acc = mts_map.get(base_key)
                mts_map[base_key] = (0 if acc is None else acc) + wi * sd[k].float()
    return lora_map, mts_map


def soup_merged(adapter_dirs, weights, base_dir: Path, out_dir: Path):
    lora_map, mts_map = collect_deltas(adapter_dirs, weights)
    print(f"[soup/merged] lora-modified tensors: {len(lora_map)}, "
          f"modules_to_save tensors: {len(mts_map)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = base_dir / "model.safetensors.index.json"
    if idx_path.exists():
        shards = sorted({v for v in json.load(open(idx_path))["weight_map"].values()})
    else:
        shards = ["model.safetensors"]

    n_lora_applied = n_mts_applied = 0
    for shard in shards:
        new_shard = {}
        with safe_open(str(base_dir / shard), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in mts_map:
                    new_shard[k] = mts_map[k].to(t.dtype)
                    n_mts_applied += 1
                elif k in lora_map:
                    acc = t.float()
                    for wi, s, A, B in lora_map[k]:
                        acc = acc + wi * s * (B @ A)
                    new_shard[k] = acc.to(t.dtype)
                    n_lora_applied += 1
                else:
                    new_shard[k] = t
        save_file(new_shard, str(out_dir / shard), metadata={"format": "pt"})
        print(f"  ↳ wrote {shard}")

    # copy config / tokenizer / preprocessor etc. (everything except weights)
    for p in base_dir.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p, out_dir / p.name)

    print(f"  applied: lora→{n_lora_applied} tensors, modules_to_save→{n_mts_applied} tensors")
    miss_l = len(lora_map) - n_lora_applied
    miss_m = len(mts_map) - n_mts_applied
    if miss_l or miss_m:
        print(f"⚠️  unmatched keys: lora {miss_l}, mts {miss_m} — check key mapping!")
    print(f"  ↳ {out_dir}  (full merged model; point inference at this dir)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="Adapter dirs (final or checkpoint-XXXX), 2+")
    ap.add_argument("--mode", choices=["adapter", "merged"], default="adapter")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Per-adapter weights (default uniform; auto-normalized)")
    ap.add_argument("--base_model", default=None,
                    help="Base model dir (required for --mode merged)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--check", action="store_true",
                    help="Only report cross-adapter lora_A cosine similarity and exit")
    args = ap.parse_args()

    if args.check:
        run_check(args.adapters)
        return
    if len(args.adapters) < 2:
        raise SystemExit("❌ need at least 2 adapters to soup")
    if not args.out_dir:
        raise SystemExit("❌ --out_dir required")

    if args.mode == "adapter":
        soup_adapter(args.adapters, args.weights, Path(args.out_dir))
    else:
        if not args.base_model:
            raise SystemExit("❌ --mode merged requires --base_model")
        soup_merged(args.adapters, args.weights, Path(args.base_model), Path(args.out_dir))


if __name__ == "__main__":
    main()
