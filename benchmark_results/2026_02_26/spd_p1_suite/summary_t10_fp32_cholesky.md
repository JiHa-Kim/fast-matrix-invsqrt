# SPD `p=1` Summary (`t10`, `fp32`, Cholesky Baseline)

Source logs:

- `t10_fp32_cholesky/solve_spd_p1_k1_t10_fp32_cholesky.txt`
- `t10_fp32_cholesky/solve_spd_p1_k16_t10_fp32_cholesky.txt`
- `t10_fp32_cholesky/solve_spd_p1_k64_t10_fp32_cholesky.txt`

## Mean Over `k in {1,16,64}` at `n=1024`

| case | method | mean ms | mean relerr vs true |
|---|---:|---:|---:|
| gaussian_spd | PE-Quad-Coupled-Apply | 4.209 | 5.469e-04 |
| gaussian_spd | Chebyshev-Apply | 5.471 | 4.502e-04 |
| gaussian_spd | PE-Quad-Inverse-Multiply | 6.870 | 5.136e-04 |
| gaussian_spd | Torch-Solve | 6.387 | 5.220e-06 |
| gaussian_spd | Torch-Cholesky-Solve | 3.240 | 2.289e-07 |
| illcond_1e6 | PE-Quad-Coupled-Apply | 3.647 | 5.813e-04 |
| illcond_1e6 | Chebyshev-Apply | 4.957 | 4.535e-04 |
| illcond_1e6 | PE-Quad-Inverse-Multiply | 6.389 | 6.517e-04 |
| illcond_1e6 | Torch-Solve | 5.680 | 1.096e-05 |
| illcond_1e6 | Torch-Cholesky-Solve | 2.571 | 2.284e-07 |

## Takeaways

- `Torch-Cholesky-Solve` is the strongest exact baseline for SPD `p=1` in this environment.
- `PE-Quad-Coupled-Apply` remains much faster than `Torch-Solve` (`~1.5x`) but slower than `Torch-Cholesky-Solve`.
- Approximate PE methods sit around `5e-4` to `7e-4` relative error, while exact baselines are near numerical precision.
