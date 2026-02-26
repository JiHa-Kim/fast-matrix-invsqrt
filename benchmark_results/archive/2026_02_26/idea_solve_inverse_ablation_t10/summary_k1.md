# Solve-Inverse Ideas Ablation

## Per-Case Results

| size | case | tag | method | precond | coeff | ms | relerr vs solve | delta ms vs A0 |
|---:|---|---|---|---|---|---:|---:|---:|
| 1024 | gaussian_shifted | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.713 | 6.452e-04 | +0.000 |
| 1024 | gaussian_shifted | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.715 | 8.288e-04 | +0.002 |
| 1024 | gaussian_shifted | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.122 | 6.423e-04 | +0.409 |
| 1024 | gaussian_shifted | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.731 | 2.000e-01 | +0.017 |
| 1024 | gaussian_shifted | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.023 | 6.452e-04 | +0.310 |
| 1024 | gaussian_shifted | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.021 | 6.452e-04 | +0.308 |
| 1024 | gaussian_shifted | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.163 | 6.452e-04 | +1.449 |
| 1024 | gaussian_shifted | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.360 | 3.847e-07 | +1.646 |
| 1024 | nonnormal_upper | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.598 | 2.717e-04 | +0.000 |
| 1024 | nonnormal_upper | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.584 | 3.483e-04 | -0.015 |
| 1024 | nonnormal_upper | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.041 | 7.214e-04 | +0.443 |
| 1024 | nonnormal_upper | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.611 | 1.999e-01 | +0.012 |
| 1024 | nonnormal_upper | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 2.880 | 2.717e-04 | +0.282 |
| 1024 | nonnormal_upper | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.059 | 2.717e-04 | +0.460 |
| 1024 | nonnormal_upper | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.155 | 2.717e-04 | +1.556 |
| 1024 | nonnormal_upper | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.338 | 1.253e-07 | +1.739 |
| 1024 | similarity_posspec | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.733 | 6.557e-04 | +0.000 |
| 1024 | similarity_posspec | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.710 | 4.553e+10 | -0.023 |
| 1024 | similarity_posspec | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.144 | 6.224e-04 | +0.411 |
| 1024 | similarity_posspec | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.728 | 1.995e-01 | -0.005 |
| 1024 | similarity_posspec | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 2.972 | 6.557e-04 | +0.238 |
| 1024 | similarity_posspec | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.031 | 6.557e-04 | +0.297 |
| 1024 | similarity_posspec | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.186 | 6.557e-04 | +1.453 |
| 1024 | similarity_posspec | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.366 | 5.143e-07 | +1.632 |
| 1024 | similarity_posspec_hard | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.710 | 9.996e-01 | +0.000 |
| 1024 | similarity_posspec_hard | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.724 | 6.592e-01 | +0.014 |
| 1024 | similarity_posspec_hard | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.136 | 9.978e-01 | +0.426 |
| 1024 | similarity_posspec_hard | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.722 | 9.996e-01 | +0.012 |
| 1024 | similarity_posspec_hard | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 7.040 | 2.757e-03 | +4.330 |
| 1024 | similarity_posspec_hard | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 5.264 | 2.757e-03 | +2.554 |
| 1024 | similarity_posspec_hard | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 5.697 | 2.757e-03 | +2.987 |
| 1024 | similarity_posspec_hard | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.342 | 2.757e-03 | +1.632 |

## Aggregated Effects

| size | tag | moderate mean ms | moderate mean relerr | hard mean ms | hard mean relerr |
|---:|---|---:|---:|---:|---:|
| 1024 | A0 | 2.682 | 5.242e-04 | 2.710 | 9.996e-01 |
| 1024 | A1 | 2.670 | 1.518e+10 | 2.724 | 6.592e-01 |
| 1024 | A2 | 3.103 | 6.620e-04 | 3.136 | 9.978e-01 |
| 1024 | A3 | 2.690 | 1.998e-01 | 2.722 | 9.996e-01 |
| 1024 | A4 | 2.958 | 5.242e-04 | 7.040 | 2.757e-03 |
| 1024 | A5 | 3.037 | 5.242e-04 | 5.264 | 2.757e-03 |
| 1024 | A6 | 4.168 | 5.242e-04 | 5.697 | 2.757e-03 |
| 1024 | A7 | 4.354 | 3.414e-07 | 4.342 | 2.757e-03 |

## Cell Definitions

- `A0`: baseline: row-norm + tuned coeff + coupled apply
- `A1`: precond only: frob scaling
- `A2`: precond only: ruiz equilibration
- `A3`: coeff only: tuned safety x1.2 (more conservative)
- `A4`: safety only: final residual fallback
- `A5`: safety only: final + early fallback
- `A6`: adaptive runtime switch + safety
- `A7`: exact reference: torch solve

