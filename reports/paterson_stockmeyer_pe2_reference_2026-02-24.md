# Paterson-Stockmeyer PE2 Reference Benchmark (2026-02-24)

## Scope

This report evaluates an experimental Paterson-Stockmeyer (PS) rewrite of the
`PE2` non-terminal update (`Y <- B Y B`) as a reference-only experiment.

- Baseline: `PE2` (current mainline implementation)
- Experimental: `PE2-PS` (PS-style polynomial split)
- Date run: February 24, 2026
- Main rigorous raw log: `artifacts/benchmarks/bench_ps_reference_rigorous_2026-02-24.txt`
- Earlier quick check log: `artifacts/benchmarks/bench_ps_reference_2026-02-24.txt`

## Command

```bash
$env:PYTHONPATH='.'
uv run python experimental/benchmark_pe2_ps_reference.py --sizes 256,512,1024 --trials 20 --warmup 4 --timing-reps 40 --dtype fp32 --power-iters 8 --mv-samples 8 --hard-probe-iters 8
```

## Result Summary

- Coverage: 15 `(size, case)` combinations (`3 sizes x 5 cases`), 20 trials each.
- `PE2-PS` was slower in 15/15 combinations.
- Median iteration latency penalty:
  - ~`+0.14 to +0.16 ms` at sizes 256/512
  - ~`+0.39 to +0.41 ms` at size 1024
- Residual/hard quality:
  - Worse in 13/15 combinations
  - Slightly better in 2/15 combinations (`1024 illcond_1e6`, `1024 spike`)
- Symmetry diagnostics (`symX`, `symW`) were worse for `PE2-PS` in all 15/15 combinations.

## Representative Samples

- Size 256, `gaussian_spd`:
  - `PE2`: `0.468 ms`, residual `4.895e-03`, hard `4.536e-03`
  - `PE2-PS`: `0.624 ms`, residual `6.959e-03`, hard `6.718e-03`
- Size 1024, `illcond_1e6` (one of the two quality-favorable PS cases):
  - `PE2`: `2.359 ms`, residual `5.569e-03`, hard `5.578e-03`
  - `PE2-PS`: `2.771 ms`, residual `5.030e-03`, hard `5.014e-03`
- Size 1024, `spike` (other quality-favorable PS case):
  - `PE2`: `2.372 ms`, residual `6.208e-03`, hard `6.225e-03`
  - `PE2-PS`: `2.770 ms`, residual `5.940e-03`, hard `5.967e-03`

## Decision

Keep PS implementation as reference only. It does not provide a speed-quality
tradeoff favorable to the current mainline objective.
