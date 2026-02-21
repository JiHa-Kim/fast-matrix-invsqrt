# Less Favorable Methods Archive
Date: 2026-02-21

This document tracks methods/paths that were tested and found less favorable for the current mainline benchmark configuration.

Baseline context:
- runner: `matrix_isqrt.py`
- dtype: `bf16`
- policy: `--auto-policy size_rho`
- timing: `--timing-reps 80`
- metrics: `--metrics-mode fast --power-iters 8 --mv-samples 8`

## Summary
| Method/Path | Outcome | Evidence | Status |
|---|---|---|---|
| `lambda_max` estimator: `power` | Slower than `row_sum` across tested AUTO cases | Prior A/B report showed AUTO delta (`power - row_sum`) from `+0.600 ms` to `+2.244 ms` with no decisive quality gain | Removed from main runner |
| No terminal final step (`--no-terminal-last-step`) | Unnecessary extra work; final `Y` update does not affect returned `X` | A/B showed residual parity and mostly better/neutral runtime with terminal mode; largest observed NS3 gain about `-1.569 ms` with terminal mode | Removed from main runner |
| Legacy raw logs (`bench_rigorous_*`, `bench_terminal_*`, `bench_rig_pow_*`) | Historical only, superseded by current baselines | Superseded by refreshed baseline artifacts | Deleted from active artifact set |

## Current Mainline (kept)
- `artifacts/benchmarks/bench_rig_row_256_512.txt`
- `artifacts/benchmarks/bench_rig_row_1024.txt`
- `reports/benchmark_report_2026-02-21.md`

## Related Commits
- `3d2d086` added optional `power` estimator path
- `a62dbc9` added terminal final-step optimization
- `a7c2027` isolated main runner to optimized defaults
- `d445e4d` pruned outdated logs from active artifacts
- `5af045d` refreshed current baselines and simplified reports
