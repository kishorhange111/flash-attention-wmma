GPU **Tesla T4**, fp16, head dim 64. TFLOP/s = attention FLOPs (causal counted as half) / median time.

SASS of `flash_fwd_kernel`: {'unpadded': {'HMMA': 128, 'FFMA': 1, 'LDS': 275, 'STS': 257}, 'padded': {'HMMA': 128, 'FFMA': 1, 'LDS': 275, 'STS': 257}}

| B×H×N | causal | wmma v2 (padded) | wmma v1 | sdpa | triton (no mma on sm75) | max |err| vs SDPA |
|---|---|---|---|---|---|---|
| 8×16×512 | False | 6.10 | 1.54 | 14.29 | 0.52 | 1.2e-04 |
| 8×16×512 | True | 5.41 | 1.34 | 12.01 | 0.13 | 4.9e-04 |
| 4×16×1024 | False | 6.61 | 1.58 | 13.73 | 0.52 | 1.2e-04 |
| 4×16×1024 | True | 6.12 | 1.46 | 12.89 | 0.13 | 4.9e-04 |
| 2×16×2048 | False | 6.70 | 1.61 | 13.48 | 0.52 | 1.2e-04 |
| 2×16×2048 | True | 6.56 | 1.53 | 12.77 | 0.14 | 4.9e-04 |
| 1×16×4096 | False | 6.75 | 1.62 | 13.00 | 0.52 | 6.1e-05 |
| 1×16×4096 | True | 6.73 | 1.57 | 12.46 | 0.14 | 2.4e-04 |
