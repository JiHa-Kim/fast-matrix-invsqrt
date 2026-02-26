# Non-SPD `p=1` Summary (`t10`, `fp32`, safe-early)

Source logs:

- `t10_fp32_safe_early/solve_nonspd_p1_k1_t10_fp32_safe_early.txt`
- `t10_fp32_safe_early/solve_nonspd_p1_k16_t10_fp32_safe_early.txt`
- `t10_fp32_safe_early/solve_nonspd_p1_k64_t10_fp32_safe_early.txt`

Config highlights:

- `--nonspd-safe-fallback-tol 0.01`
- `--nonspd-safe-early-y-tol 0.8`

## Mean Over `k in {1,16,64}` at `n=1024`

| case | method | mean ms | mean relerr vs solve |
|---|---:|---:|---:|
| gaussian_shifted | PE-Quad-Coupled-Apply | 2.839 | 7.043e-04 |
| gaussian_shifted | PE-Quad-Coupled-Apply-Safe | 3.199 | 7.043e-04 |
| gaussian_shifted | PE-Quad-Coupled-Apply-Adaptive | 4.310 | 7.043e-04 |
| gaussian_shifted | PE-Quad-Inverse-Multiply | 4.191 | 6.082e-04 |
| gaussian_shifted | Torch-Solve | 4.653 | 5.597e-05 |
| nonnormal_upper | PE-Quad-Coupled-Apply | 2.639 | 5.645e-04 |
| nonnormal_upper | PE-Quad-Coupled-Apply-Safe | 2.976 | 5.645e-04 |
| nonnormal_upper | PE-Quad-Coupled-Apply-Adaptive | 4.177 | 5.645e-04 |
| nonnormal_upper | PE-Quad-Inverse-Multiply | 3.935 | 4.770e-04 |
| nonnormal_upper | Torch-Solve | 7.324 | 4.917e-05 |
| similarity_posspec | PE-Quad-Coupled-Apply | 2.776 | 7.222e-04 |
| similarity_posspec | PE-Quad-Coupled-Apply-Safe | 3.110 | 7.222e-04 |
| similarity_posspec | PE-Quad-Coupled-Apply-Adaptive | 4.244 | 7.222e-04 |
| similarity_posspec | PE-Quad-Inverse-Multiply | 4.139 | 7.446e-04 |
| similarity_posspec | Torch-Solve | 4.554 | 6.834e-05 |
| similarity_posspec_hard | PE-Quad-Coupled-Apply | 2.755 | 9.988e-01 |
| similarity_posspec_hard | PE-Quad-Coupled-Apply-Safe | 5.483 | 6.473e-03 |
| similarity_posspec_hard | PE-Quad-Coupled-Apply-Adaptive | 5.983 | 6.473e-03 |
| similarity_posspec_hard | PE-Quad-Inverse-Multiply | 4.108 | 9.988e-01 |
| similarity_posspec_hard | Torch-Solve | 4.557 | 6.473e-03 |

## Takeaways

- Approximate throughput path: `PE-Quad-Coupled-Apply` remains fastest on moderate non-SPD cases.
- Robust path: `PE-Quad-Coupled-Apply-Safe` recovers solve-level stability on `similarity_posspec_hard` and is consistently faster than adaptive mode.
- Hard-case latency: safe mode is still slower than `Torch-Solve` on the hardest case because exact fallback is triggered, but catastrophic divergence is removed.
