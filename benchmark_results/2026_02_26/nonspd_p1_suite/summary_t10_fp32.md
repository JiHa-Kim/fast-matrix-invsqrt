# Non-SPD `p=1` Summary (`t10`, `fp32`)

Source logs:

- `t10_fp32/solve_nonspd_p1_k1_t10_fp32.txt`
- `t10_fp32/solve_nonspd_p1_k16_t10_fp32.txt`
- `t10_fp32/solve_nonspd_p1_k64_t10_fp32.txt`

## Mean Over `k in {1,16,64}` at `n=1024`

| case | method | mean ms | mean relerr vs solve |
|---|---:|---:|---:|
| gaussian_shifted | PE-Quad-Coupled-Apply | 2.843 | 7.043e-04 |
| gaussian_shifted | PE-Quad-Coupled-Apply-Adaptive | 4.223 | 7.043e-04 |
| gaussian_shifted | PE-Quad-Inverse-Multiply | 4.186 | 6.082e-04 |
| gaussian_shifted | Torch-Solve | 4.650 | 5.597e-05 |
| nonnormal_upper | PE-Quad-Coupled-Apply | 2.651 | 5.645e-04 |
| nonnormal_upper | PE-Quad-Coupled-Apply-Adaptive | 4.022 | 5.645e-04 |
| nonnormal_upper | PE-Quad-Inverse-Multiply | 3.948 | 4.770e-04 |
| nonnormal_upper | Torch-Solve | 4.552 | 4.917e-05 |
| similarity_posspec | PE-Quad-Coupled-Apply | 2.757 | 7.222e-04 |
| similarity_posspec | PE-Quad-Coupled-Apply-Adaptive | 4.076 | 7.222e-04 |
| similarity_posspec | PE-Quad-Inverse-Multiply | 4.127 | 7.446e-04 |
| similarity_posspec | Torch-Solve | 4.575 | 6.834e-05 |
| similarity_posspec_hard | PE-Quad-Coupled-Apply | 2.755 | 9.988e-01 |
| similarity_posspec_hard | PE-Quad-Coupled-Apply-Adaptive | 17.203 | 0.000e+00 |
| similarity_posspec_hard | PE-Quad-Inverse-Multiply | 4.116 | 9.988e-01 |
| similarity_posspec_hard | Torch-Solve | 4.569 | 6.473e-03 |

## Takeaway

- For moderate non-SPD cases, `PE-Quad-Coupled-Apply` is consistently fastest.
- For hard non-normal cases, fast PE variants can fail badly (`~1.0` relative error).
- Adaptive mode with `--nonspd-safe-fallback-tol 0.01` restores robustness by falling back to exact solve when needed, at significantly higher latency.

