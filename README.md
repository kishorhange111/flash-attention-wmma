# flash-attention-wmma

[![CI](https://github.com/kishorhange111/flash-attention-wmma/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/flash-attention-wmma/actions/workflows/ci.yml)

A FlashAttention-2 forward kernel in CUDA C++ that runs its two matmuls on **tensor cores**
through the WMMA API, written for the Turing T4 (sm_75).

**Why.** In [triton-kernels-from-scratch](https://github.com/kishorhange111/triton-kernels-from-scratch)
the PTX of the Triton FlashAttention kernel compiled for sm_75 contained **no `mma`
instructions**, only about 2,000 fp32 FMAs. `tl.dot` fell back to CUDA cores on this GPU,
so even its best tile configuration reached 1.35 TFLOP/s against SDPA's 12. This repo writes
the same algorithm one level lower, where tensor-core use is explicit and can be checked in
the SASS. **Result: 6.7 TFLOP/s, 5× the best Triton configuration and about half of SDPA.**

## Kernel ([`flash_fwd.cu`](src/fawmma/flash_fwd.cu))

One thread block holds 64 query rows for one (batch, head), and each of its 4 warps owns 16
rows. Each warp keeps its Q fragments in registers. For every 64-row K/V tile:

1. K and V are copied to shared memory with 16-byte vector loads.
2. `S = Q Kᵀ`: 4 output fragments × 4 k-steps of 16×16×16 `mma_sync`. Kᵀ needs no
   transpose; K stored row-major *is* Kᵀ in column-major layout.
3. Online softmax in shared memory, with two lanes per row and one shuffle for the max and
   sum. The running max and denominator are updated, `P = exp2(S − m)` is written as fp16
   over the K tile (no longer needed), and the warp's rows of the fp32 O accumulator are
   rescaled.
4. `O += P V`: each O fragment is loaded as the accumulator, gets 4 more `mma_sync`, and is
   stored back.

The epilogue writes `O / l` and the log-sum-exp.

**Bank conflicts (v1 → v2).** Shared memory has 32 four-byte banks. In v1 every tile row was
64 elements wide, so row r of an fp32 tile started at bank `64r mod 32 = 0`. In the softmax
and rescale loops, each lane walks its own row, so the 16 rows a warp touches all hit the
same bank: a 16-way conflict on every access. v2 pads fp32 rows to 68 floats and fp16 rows
to 72 halves, the smallest strides WMMA accepts (`ldm` a multiple of 16 bytes) that move each
row by 4 banks, and has the two lanes of a row take even and odd columns. That leaves 2-way
conflicts. Both layouts are compiled from one template, and `padded=False` selects v1. Causal attention skips key tiles past the
diagonal and masks only on the diagonal tile. Shared memory is 48.5 KiB (v1) or 52.5 KiB (v2),
which needs the opt-in above the 48 KiB default. The extension is built at first use with
`torch.utils.cpp_extension`.

## Tests

The GPU tests run on Colab. They check that:
- output and LSE match an fp32 reference for several shapes, causal and not (fp16
  tolerance);
- scores of ±500 don't overflow;
- unsupported shapes are rejected.

CI runs on CPU, where it checks only the reference and skips the kernel tests.

```
pytest -q                            # on a CUDA machine
python -m fawmma.bench --out results # TFLOP/s vs SDPA (and Triton), plus HMMA/FFMA counts in SASS
```

## Results (Tesla T4)

These are 19 GPU tests, for both layouts, all passing on Colab. Throughput is in TFLOP/s:
attention FLOPs (causal counted as half) divided by the median of 5 `do_bench` rounds. The
GPU was otherwise idle. Raw numbers are in [`results/`](results).

| B×H×N | causal | **WMMA v2** | WMMA v1 | SDPA | Triton (default 64×64) |
|---|---|---|---|---|---|
| 8×16×512 | – | **6.10** | 1.54 | 14.29 | 0.52 |
| 8×16×512 | ✓ | **5.41** | 1.34 | 12.01 | 0.13 |
| 4×16×1024 | – | **6.61** | 1.58 | 13.73 | 0.52 |
| 4×16×1024 | ✓ | **6.12** | 1.46 | 12.89 | 0.13 |
| 2×16×2048 | – | **6.70** | 1.61 | 13.48 | 0.52 |
| 2×16×2048 | ✓ | **6.56** | 1.53 | 12.77 | 0.14 |
| 1×16×4096 | – | **6.75** | 1.62 | 13.00 | 0.52 |
| 1×16×4096 | ✓ | **6.73** | 1.57 | 12.46 | 0.14 |

The maximum |error| against SDPA is ≤ 4.9e-4 everywhere, which is fp16 rounding.

- **Tensor cores are used.** The SASS of each kernel has **128 `HMMA`** instructions and a
  single `FFMA`. On sm_75 each 16×16×16 `mma_sync` is 4 `HMMA.1688` (m16n8k8), and a warp
  issues 32 `mma_sync` per K/V tile (16 for S, 16 for P·V). The Triton kernel's PTX had no `mma` at all.
- **Padding is worth 4.0–4.3×** (1.6 → 6.7 TFLOP/s). The instruction counts of v1 and v2
  are identical, with 275 `LDS` and 257 `STS` each. The whole difference is bank conflicts:
  the same instructions, replayed up to 16 times.
- **WMMA v2 reaches 43–54% of SDPA.** It is 5× the best tuned Triton configuration (1.35)
  and 12–13× the default one (non-causal). It does better at longer sequences, where per-tile overhead is
  amortised.
- **Causal costs almost nothing per useful FLOP** (6.56 against 6.70 at N = 2048). Fully
  masked tiles are skipped, and only the diagonal tile pays for the mask.

**What is left on the table.** With 52.5 KiB of shared memory per block, a T4 SM (64 KiB)
fits one 4-warp block. That is too few warps to hide the latency of shared-memory traffic
and global loads. The next steps:
- keep S and O in registers instead of shared memory. This needs `mma.sync` PTX, whose
  fragment layout is documented; WMMA's layout is opaque, so per-row softmax and rescaling
  have to go through shared memory;
- double-buffer K/V with `cp.async`, which needs sm_80 (Turing has no async copy);
- use 128-row query tiles with 8 warps.

The first alone leaves only K/P and V in shared memory (18.5 KiB), which is room for three
blocks per SM if registers allow.
