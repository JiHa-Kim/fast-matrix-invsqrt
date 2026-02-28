# Spectral Convergence Benchmark

Generated: 2026-02-28T04:35:06

## Run Configuration

- coeff_mode: `precomputed`
- coeff_no_final_safety: `False`
- coeff_safety: `1.0`
- coeff_seed: `0`
- device: `cuda`
- dtype: `fp64`
- integrity_checksums: `True`
- json_out: ``
- l_target: `0.05`
- manifest_out: ``
- n: `1024`
- out: ``
- p: `2`
- pe_steps: `4`
- prod: `True`
- run_name: `spectral_convergence`
- seed: `1234`
- trials: `10`

## Column Definitions

- **Min eig / Max eig**: Minimum and maximum eigenvalues of the current iterate.
- **rho(I-Y)**: Spectral radius of the residual matrix, $\rho(I - Y) = \max_i |1 - \lambda_i|$. Measures overall closeness to identity.
- **log(M/m)**: Log-width of the spectral interval, $\log(\lambda_{\max}/\lambda_{\min})$. Primary indicator of iteration progress for coupled methods.
- **C90% / C99%**: Fraction of eigenvalues clustered within 10% and 1% of identity (1.0).

## Coefficients (PE-Quad)

| Step | a | b | c |
| ---: | ---: | ---: | ---: |
| 0 | 3.9021 | -7.5907 | 4.8608 |
| 1 | 1.9378 | -1.3493 | 0.4110 |
| 2 | 1.8751 | -1.2502 | 0.3751 |
| 3 | 1.8750 | -1.2499 | 0.3750 |

## PE-Quad (Worst Case Over Trials)

| Step | Min eig | Max eig | rho(I-Y) | log(M/m) | C90% | C99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 9.50e-01 | 2.996 | 10.5% | 1.1% |
| 1 | 0.6247 | 1.3742 | 3.75e-01 | 0.788 | 19.0% | 2.0% |
| 2 | 0.9843 | 1.0157 | 1.57e-02 | 0.031 | 100.0% | 99.2% |
| 3 | 1.0000 | 1.0000 | 1.06e-06 | 0.000 | 100.0% | 100.0% |
| 4 | 1.0000 | 1.0000 | 5.96e-08 | 0.000 | 100.0% | 100.0% |

## Newton-Schulz (Worst Case Over Trials)

| Step | Min eig | Max eig | rho(I-Y) | log(M/m) | C90% | C99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 9.50e-01 | 2.996 | 10.5% | 1.1% |
| 1 | 0.1088 | 1.0000 | 8.91e-01 | 2.218 | 36.4% | 12.0% |
| 2 | 0.2273 | 1.0000 | 7.73e-01 | 1.481 | 65.0% | 38.7% |
| 3 | 0.4369 | 1.0000 | 5.63e-01 | 0.828 | 84.9% | 66.9% |
| 4 | 0.7176 | 1.0000 | 2.82e-01 | 0.332 | 95.6% | 85.8% |

## Reproducibility

This report is paired with:
- `benchmark_results/runs/2026_02_28/043454_spectral_convergence/spectral_convergence.json` (raw per-step rows)
- `benchmark_results/runs/2026_02_28/043454_spectral_convergence/spectral_manifest.json` (run metadata + reproducibility fingerprint)
- `.sha256` sidecars for all output files
