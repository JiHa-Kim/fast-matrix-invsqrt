# Solve-Inverse Ideas Ablation

## Per-Case Results

| size | case | tag | method | precond | coeff | ms | relerr vs solve | delta ms vs A0 |
|---:|---|---|---|---|---|---:|---:|---:|
| 1024 | gaussian_shifted | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.753 | 7.251e-04 | +0.000 |
| 1024 | gaussian_shifted | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.805 | 8.207e-04 | +0.051 |
| 1024 | gaussian_shifted | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.203 | 7.135e-04 | +0.450 |
| 1024 | gaussian_shifted | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.765 | 2.000e-01 | +0.012 |
| 1024 | gaussian_shifted | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.025 | 7.251e-04 | +0.272 |
| 1024 | gaussian_shifted | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.047 | 7.251e-04 | +0.294 |
| 1024 | gaussian_shifted | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.931 | 7.251e-04 | +2.177 |
| 1024 | gaussian_shifted | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.607 | 9.222e-05 | +1.854 |
| 1024 | nonnormal_upper | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.658 | 4.545e-04 | +0.000 |
| 1024 | nonnormal_upper | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.659 | 4.594e-04 | +0.001 |
| 1024 | nonnormal_upper | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.115 | 7.831e-04 | +0.458 |
| 1024 | nonnormal_upper | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.659 | 1.999e-01 | +0.001 |
| 1024 | nonnormal_upper | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 2.951 | 4.545e-04 | +0.294 |
| 1024 | nonnormal_upper | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.010 | 4.545e-04 | +0.353 |
| 1024 | nonnormal_upper | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.080 | 4.545e-04 | +1.422 |
| 1024 | nonnormal_upper | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.678 | 8.126e-05 | +2.020 |
| 1024 | similarity_posspec | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.773 | 7.531e-04 | +0.000 |
| 1024 | similarity_posspec | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.773 | 4.548e+10 | +0.001 |
| 1024 | similarity_posspec | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.314 | 7.149e-04 | +0.541 |
| 1024 | similarity_posspec | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.771 | 1.995e-01 | -0.002 |
| 1024 | similarity_posspec | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.088 | 7.531e-04 | +0.315 |
| 1024 | similarity_posspec | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 3.105 | 7.531e-04 | +0.332 |
| 1024 | similarity_posspec | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 4.168 | 7.531e-04 | +1.395 |
| 1024 | similarity_posspec | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.631 | 1.137e-04 | +1.858 |
| 1024 | similarity_posspec_hard | A0 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.00 | 2.794 | 9.995e-01 | +0.000 |
| 1024 | similarity_posspec_hard | A1 | PE-Quad-Coupled-Apply | frob | tuned@x1.00 | 2.765 | 6.588e-01 | -0.030 |
| 1024 | similarity_posspec_hard | A2 | PE-Quad-Coupled-Apply | ruiz | tuned@x1.00 | 3.221 | 9.983e-01 | +0.427 |
| 1024 | similarity_posspec_hard | A3 | PE-Quad-Coupled-Apply | row-norm | tuned@x1.20 | 2.801 | 9.996e-01 | +0.007 |
| 1024 | similarity_posspec_hard | A4 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 7.443 | 1.035e-02 | +4.649 |
| 1024 | similarity_posspec_hard | A5 | PE-Quad-Coupled-Apply-Safe | row-norm | tuned@x1.00 | 5.595 | 1.035e-02 | +2.800 |
| 1024 | similarity_posspec_hard | A6 | PE-Quad-Coupled-Apply-Adaptive | row-norm | tuned@x1.00 | 6.124 | 1.035e-02 | +3.330 |
| 1024 | similarity_posspec_hard | A7 | Torch-Solve | row-norm | tuned@x1.00 | 4.684 | 1.035e-02 | +1.890 |

## Aggregated Effects

| size | tag | moderate mean ms | moderate mean relerr | hard mean ms | hard mean relerr |
|---:|---|---:|---:|---:|---:|
| 1024 | A0 | 2.728 | 6.442e-04 | 2.794 | 9.995e-01 |
| 1024 | A1 | 2.746 | 1.516e+10 | 2.765 | 6.588e-01 |
| 1024 | A2 | 3.211 | 7.372e-04 | 3.221 | 9.983e-01 |
| 1024 | A3 | 2.732 | 1.998e-01 | 2.801 | 9.996e-01 |
| 1024 | A4 | 3.021 | 6.442e-04 | 7.443 | 1.035e-02 |
| 1024 | A5 | 3.054 | 6.442e-04 | 5.595 | 1.035e-02 |
| 1024 | A6 | 4.393 | 6.442e-04 | 6.124 | 1.035e-02 |
| 1024 | A7 | 4.639 | 9.574e-05 | 4.684 | 1.035e-02 |

## Cell Definitions

- `A0`: baseline: row-norm + tuned coeff + coupled apply
- `A1`: precond only: frob scaling
- `A2`: precond only: ruiz equilibration
- `A3`: coeff only: tuned safety x1.2 (more conservative)
- `A4`: safety only: final residual fallback
- `A5`: safety only: final + early fallback
- `A6`: adaptive runtime switch + safety
- `A7`: exact reference: torch solve

