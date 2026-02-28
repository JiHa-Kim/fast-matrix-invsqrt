# Spectral Convergence Benchmark

Generated: 2026-02-28T00:41:49

## Run Configuration

- n: `128`
- p: `2`
- trials: `5`
- l_target: `0.05`
- dtype: `fp64`
- device: `cuda`
- seed: `1234`
- coeff_mode: `precomputed`
- coeff_seed: `0`
- coeff_safety: `1.0`
- coeff_no_final_safety: `False`
- pe_steps: `4`

## Coefficients (PE-Quad)

| Step | a | b | c |
|---:|---:|---:|---:|
| 0 | 3.902148485 | -7.590706825 | 4.860831261 |
| 1 | 1.937780857 | -1.349293113 | 0.410987377 |
| 2 | 1.875123501 | -1.250201106 | 0.375077546 |
| 3 | 1.874953985 | -1.249907970 | 0.374954015 |

## PE-Quad (Worst Case Over Trials)

| Step | Min λ | Max λ | Mean λ | ρ(I-Y) | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.9% | 1.6% |
| 1 | 0.6247 | 1.3742 | 0.9661 | 3.75e-01 | 18.8% | 1.6% |
| 2 | 0.9843 | 1.0157 | 1.0000 | 1.57e-02 | 100.0% | 98.4% |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.06e-06 | 100.0% | 100.0% |
| 4 | 1.0000 | 1.0000 | 1.0000 | 5.96e-08 | 100.0% | 100.0% |

## Newton-Schulz (Worst Case Over Trials)

| Step | Min λ | Max λ | Mean λ | ρ(I-Y) | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.9% | 1.6% |
| 1 | 0.1088 | 1.0000 | 0.7195 | 8.91e-01 | 36.7% | 12.5% |
| 2 | 0.2273 | 1.0000 | 0.8650 | 7.73e-01 | 64.8% | 39.1% |
| 3 | 0.4369 | 1.0000 | 0.9493 | 5.63e-01 | 84.4% | 66.4% |
| 4 | 0.7176 | 1.0000 | 0.9870 | 2.82e-01 | 95.3% | 85.9% |

## Reproducibility

This report is paired with:
- `spectral_convergence.json` (raw per-step rows)
- `spectral_manifest.json` (run metadata + reproducibility fingerprint)
- `.sha256` sidecars for all output files
