# SPD `p=1` Comprehensive Summary (`n=1024`, `k=16`, `t10`, `fp32`)

Source log:

- `comprehensive_k16_t10_fp32_cholesky/solve_spd_p1_n1024_k16_t10_all_cases_fp32_cholesky.txt`

Cases:

- `gaussian_spd`
- `illcond_1e6`
- `illcond_1e12`
- `near_rank_def`
- `spike`

## Key Methods

| case | PE-Quad-Coupled-Apply ms | Torch-Solve ms | Torch-Cholesky-Solve ms | coupled relerr | cholesky relerr |
|---|---:|---:|---:|---:|---:|
| gaussian_spd | 3.834 | 6.052 | 2.938 | 6.693e-04 | 1.679e-07 |
| illcond_1e6 | 4.352 | 6.426 | 3.332 | 7.582e-04 | 1.701e-07 |
| illcond_1e12 | 3.763 | 6.187 | 3.444 | 6.496e-04 | 1.694e-07 |
| near_rank_def | 5.440 | 7.459 | 4.289 | 6.414e-04 | 1.715e-07 |
| spike | 3.816 | 5.942 | 2.769 | 6.509e-04 | 1.847e-07 |

## Takeaways

- `PE-Quad-Coupled-Apply` is consistently faster than `Torch-Solve` across all five SPD cases.
- `Torch-Cholesky-Solve` remains faster than coupled PE in this SPD `p=1` setting while also delivering exact-solve accuracy.
- No catastrophic accuracy failures were observed in this SPD sweep for either coupled PE or exact baselines.
