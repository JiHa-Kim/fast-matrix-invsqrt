# Benchmark Report (Current Baseline)

Date: 2026-02-21

This report is generated from the current kept baseline artifacts only.

## Source Files
- `artifacts/benchmarks/bench_rig_row_256_512.txt`
- `artifacts/benchmarks/bench_rig_row_1024.txt`

## Configuration
- dtype: `bf16`
- policy: `--auto-policy size_rho`
- timing: `--timing-reps 80`
- metrics: `--metrics-mode fast --power-iters 8 --mv-samples 8`
- sizes/trials: `256,512` (`trials=30`, `warmup=4`), `1024` (`trials=20`, `warmup=3`)
- normalization: row-sum baseline (main runner default)
- terminal final step: enabled (main runner default)

## Winner Per Case (NS3 vs PE-NS3 vs PE2)
| Size | Case | Winner | ms | p95 resid |
|---:|---|---|---:|---:|
| 256 | gaussian_spd | PE2 | 1.913 | 1.078e-02 |
| 256 | illcond_1e12 | PE2 | 1.843 | 8.891e-03 |
| 256 | illcond_1e6 | PE2 | 1.935 | 9.518e-03 |
| 256 | near_rank_def | PE2 | 1.880 | 8.632e-03 |
| 256 | spike | PE2 | 1.945 | 5.182e-03 |
| 512 | gaussian_spd | PE2 | 1.908 | 9.995e-03 |
| 512 | illcond_1e12 | PE2 | 2.028 | 2.546e-03 |
| 512 | illcond_1e6 | PE2 | 1.909 | 4.772e-03 |
| 512 | near_rank_def | NS3 | 1.961 | 1.432e-02 |
| 512 | spike | PE2 | 1.949 | 4.890e-03 |
| 1024 | gaussian_spd | PE2 | 2.805 | 8.501e-03 |
| 1024 | illcond_1e12 | PE2 | 2.793 | 9.639e-04 |
| 1024 | illcond_1e6 | PE2 | 2.733 | 5.401e-04 |
| 1024 | near_rank_def | PE2 | 2.778 | 8.589e-04 |
| 1024 | spike | PE2 | 2.816 | 6.418e-03 |

## Harness Best Lines (includes AUTO)
| Size | Case | Best |
|---:|---|---|
| 256 | gaussian_spd | `PE2 @ 1.913 ms, resid=9.721e-03` |
| 256 | illcond_1e12 | `PE2 @ 1.843 ms, resid=7.979e-03` |
| 256 | illcond_1e6 | `PE2 @ 1.935 ms, resid=9.202e-03` |
| 256 | near_rank_def | `PE2 @ 1.880 ms, resid=7.826e-03` |
| 256 | spike | `PE2 @ 1.945 ms, resid=4.257e-03` |
| 512 | gaussian_spd | `PE2 @ 1.908 ms, resid=9.172e-03` |
| 512 | illcond_1e12 | `AUTO @ 2.028 ms, resid=4.934e-03` |
| 512 | illcond_1e6 | `PE2 @ 1.909 ms, resid=3.422e-03` |
| 512 | near_rank_def | `PE-NS3 @ 2.018 ms, resid=7.910e-03` |
| 512 | spike | `PE2 @ 1.949 ms, resid=4.531e-03` |
| 1024 | gaussian_spd | `PE2 @ 2.805 ms, resid=6.801e-03` |
| 1024 | illcond_1e12 | `AUTO @ 2.713 ms, resid=7.442e-04` |
| 1024 | illcond_1e6 | `PE2 @ 2.733 ms, resid=5.318e-04` |
| 1024 | near_rank_def | `AUTO @ 2.776 ms, resid=8.081e-04` |
| 1024 | spike | `AUTO @ 2.767 ms, resid=6.015e-03` |

## Notes
- Legacy/experimental benchmark logs have been removed from `artifacts/benchmarks/` to keep only current baselines.
- If the algorithm or defaults change, rerun the two baseline commands and regenerate this report.
