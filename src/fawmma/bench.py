"""Benchmark the WMMA kernel against SDPA (and, if installed, the Triton kernel from
triton-kernels-from-scratch), and check its SASS for tensor-core instructions.

    python -m fawmma.bench --out results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import torch
import torch.nn.functional as F

from . import extension_path, flash_attention


def sass_counts() -> dict:
    """Count HMMA (tensor core) and FFMA (CUDA core) instructions in the compiled kernel."""
    try:
        sass = subprocess.run(
            ["cuobjdump", "-sass", extension_path()], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return {"error": str(e)}
    lines = sass.splitlines()  # the extension contains this one kernel
    return {
        "HMMA": sum("HMMA" in ln for ln in lines),
        "FFMA": sum("FFMA" in ln for ln in lines),
        "LDS": sum(" LDS" in ln for ln in lines),
    }


def _impls(q, k, v, causal, triton_fa):
    impls = {
        "wmma": lambda: flash_attention(q, k, v, causal),
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
    }
    if triton_fa is not None:
        impls["triton (no mma on sm75)"] = lambda: triton_fa(q, k, v, causal)
    return impls


def _ms(fn, rounds=5):
    import triton.testing

    t = sorted(triton.testing.do_bench(fn, warmup=25, rep=100) for _ in range(rounds))
    return t[len(t) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.cuda.get_device_name()
    rows = []
    try:
        from tkern.flash_attn import flash_attention as triton_fa
    except ImportError:
        triton_fa = None
    for b, h, n in ((8, 16, 512), (4, 16, 1024), (2, 16, 2048), (1, 16, 4096)):
        q, k, v = (torch.randn(b, h, n, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        for causal in (False, True):
            flops = 4 * b * h * n * n * 64 * (0.5 if causal else 1.0)
            r = {"shape": f"{b}×{h}×{n}", "causal": causal}
            for name, fn in _impls(q, k, v, causal, triton_fa).items():
                r[name] = flops / _ms(fn) / 1e9  # TFLOP/s
            err = (
                (
                    flash_attention(q, k, v, causal).float()
                    - F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()
                )
                .abs()
                .max()
            )
            r["max_abs_err_vs_sdpa"] = float(err)
            rows.append(r)
            print(json.dumps(r), flush=True)
    sass = sass_counts()
    names = [k for k in rows[0] if k not in ("shape", "causal", "max_abs_err_vs_sdpa")]
    lines = [
        f"GPU **{dev}**, fp16, head dim 64. TFLOP/s = attention FLOPs (causal counted as half)"
        " / median time.",
        "",
        f"SASS of `flash_fwd_kernel`: {sass}",
        "",
        "| B×H×N | causal | " + " | ".join(names) + " | max |err| vs SDPA |",
        "|---|---|" + "---|" * (len(names) + 1),
    ]
    for r in rows:
        lines.append(
            f"| {r['shape']} | {r['causal']} | "
            + " | ".join(f"{r[k]:.2f}" for k in names)
            + f" | {r['max_abs_err_vs_sdpa']:.1e} |"
        )
    text = "\n".join(lines) + "\n"
    tag = dev.replace(" ", "_")
    with open(os.path.join(a.out, f"bench_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.out, f"bench_{tag}.json"), "w") as f:
        json.dump({"rows": rows, "sass": sass}, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
