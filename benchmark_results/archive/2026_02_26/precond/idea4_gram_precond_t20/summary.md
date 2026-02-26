# Gram Precondition Path Check

- device: cuda
- dtype: torch.bfloat16
- shape: G in R^(4096 x 1024)
- trials: 20 (alternating run order, with warmup)

| path | mean_ms | delta_vs_gram_colnorm |
|---|---:|---:|
| precond_gram_spd(col-norm, mode=none) | 2.891 | +0.00% |
| (G^T G) then precond_spd(jacobi) | 2.678 | -7.38% |

| parity | value |
|---|---:|
| mean rel Fro diff(A_norm) | 0.000e+00 |
| max rel Fro diff(A_norm) | 0.000e+00 |

