# Solve-Inverse Ideas Ablation

## Per-Case Results

| size | case | tag | method | precond | coeff | ms | relerr vs solve | delta ms vs A0 |
|---:|---|---|---|---|---|---:|---:|---:|
| 1024 | gaussian_shifted | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.821 | 7.205e-04 | +0.000 |
| 1024 | gaussian_shifted | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.832 | 8.221e-04 | +0.011 |
| 1024 | gaussian_shifted | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.238 | 7.172e-04 | +0.416 |
| 1024 | gaussian_shifted | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.837 | 2.000e-01 | +0.015 |
| 1024 | gaussian_shifted | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.089 | 7.205e-04 | +0.268 |
| 1024 | gaussian_shifted | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.118 | 7.205e-04 | +0.297 |
| 1024 | gaussian_shifted | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.177 | 7.205e-04 | +1.356 |
| 1024 | gaussian_shifted | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.700 | 7.484e-05 | +1.879 |
| 1024 | nonnormal_upper | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.659 | 4.626e-04 | +0.000 |
| 1024 | nonnormal_upper | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.677 | 4.606e-04 | +0.018 |
| 1024 | nonnormal_upper | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.106 | 7.878e-04 | +0.447 |
| 1024 | nonnormal_upper | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.680 | 1.999e-01 | +0.021 |
| 1024 | nonnormal_upper | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 2.969 | 4.626e-04 | +0.310 |
| 1024 | nonnormal_upper | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.094 | 4.626e-04 | +0.435 |
| 1024 | nonnormal_upper | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.070 | 4.626e-04 | +1.411 |
| 1024 | nonnormal_upper | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.623 | 6.581e-05 | +1.964 |
| 1024 | similarity_posspec | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.784 | 7.498e-04 | +0.000 |
| 1024 | similarity_posspec | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.786 | 4.593e+10 | +0.003 |
| 1024 | similarity_posspec | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.212 | 7.134e-04 | +0.429 |
| 1024 | similarity_posspec | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.790 | 1.995e-01 | +0.006 |
| 1024 | similarity_posspec | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.017 | 7.498e-04 | +0.234 |
| 1024 | similarity_posspec | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.046 | 7.498e-04 | +0.263 |
| 1024 | similarity_posspec | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.236 | 7.498e-04 | +1.453 |
| 1024 | similarity_posspec | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.641 | 9.061e-05 | +1.857 |
| 1024 | similarity_posspec_hard | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.791 | 9.995e-01 | +0.000 |
| 1024 | similarity_posspec_hard | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.790 | 6.667e-01 | -0.001 |
| 1024 | similarity_posspec_hard | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.222 | 9.988e-01 | +0.431 |
| 1024 | similarity_posspec_hard | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.817 | 9.996e-01 | +0.026 |
| 1024 | similarity_posspec_hard | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 7.367 | 8.477e-03 | +4.576 |
| 1024 | similarity_posspec_hard | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 5.561 | 8.477e-03 | +2.770 |
| 1024 | similarity_posspec_hard | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 6.015 | 8.477e-03 | +3.224 |
| 1024 | similarity_posspec_hard | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.612 | 8.477e-03 | +1.821 |

## Aggregated Effects

| size | tag | moderate mean ms | moderate mean relerr | hard mean ms | hard mean relerr |
|---:|---|---:|---:|---:|---:|
| 1024 | A0 | 2.755 | 6.443e-04 | 2.791 | 9.995e-01 |
| 1024 | A1 | 2.765 | 1.531e+10 | 2.790 | 6.667e-01 |
| 1024 | A2 | 3.185 | 7.395e-04 | 3.222 | 9.988e-01 |
| 1024 | A3 | 2.769 | 1.998e-01 | 2.817 | 9.996e-01 |
| 1024 | A4 | 3.025 | 6.443e-04 | 7.367 | 8.477e-03 |
| 1024 | A5 | 3.086 | 6.443e-04 | 5.561 | 8.477e-03 |
| 1024 | A6 | 4.161 | 6.443e-04 | 6.015 | 8.477e-03 |
| 1024 | A7 | 4.655 | 7.709e-05 | 4.612 | 8.477e-03 |

## Cell Definitions

- `A0`: baseline: row-norm + tuned coeff + coupled apply
- `A1`: precond only: frob scaling
- `A2`: precond only: ruiz equilibration
- `A3`: coeff only: tuned safety x1.2 (more conservative)
- `A4`: safety only: final residual fallback
- `A5`: safety only: final + early fallback
- `A6`: adaptive runtime switch + safety
- `A7`: exact reference: torch solve

