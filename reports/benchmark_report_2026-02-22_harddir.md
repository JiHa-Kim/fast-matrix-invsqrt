# Benchmark Report (2026-02-22): Hard-Direction Benchmarking Update

## Scope

This report reflects the benchmarking-only update:

- add hard-direction residual probe (`hard_dir`)
- allow target selection by metric (`--target-metric hard_dir`)
- keep algorithms unchanged

## Command

```bash
.\.venv\Scripts\python.exe matrix_isqrt.py \
  --sizes 256,512 \
  --trials 20 \
  --warmup 3 \
  --timing-reps 12 \
  --metrics-mode full \
  --power-iters 8 \
  --mv-samples 8 \
  --hard-probe-iters 8 \
  --target-metric hard_dir \
  --target-resid 0.01 \
  --dtype bf16 \
  --coeff-mode precomputed
```

Raw output:

- `artifacts/benchmarks/bench_harddir_2026-02-22.txt`

## Key Findings

1. Hard-direction metric is informative and differs from Frobenius residual ranking.
2. For difficult spectra (ill-conditioned / near-rank-def / spike), `PE2` is typically best under `hard_dir <= 1e-2`.
3. `AUTO` with current hybrid policy is often not the fastest method that satisfies hard-direction target.

## Representative Results

### Size 256, illcond_1e12 (bf16)

- `PE2`: `2.026 ms`, `hard=7.429e-03`, `resid=7.155e-03`
- `AUTO`: `2.151 ms`, `hard=9.528e-03`, `resid=9.160e-03`
- Best under hard target: `PE2`

### Size 512, near_rank_def (bf16)

- `PE2`: `1.938 ms`, `hard=3.108e-03`, `resid=3.302e-03`
- `AUTO`: `2.082 ms`, `hard=8.029e-03`, `resid=7.867e-03`
- Best under hard target: `PE2`

### Size 512, spike (bf16)

- `PE2`: `2.043 ms`, `hard=4.418e-03`, `resid=4.596e-03`
- `AUTO`: `2.174 ms`, `hard=6.874e-03`, `resid=6.842e-03`
- Best under hard target: `PE2`

## Practical Guidance

- If preconditioning quality is judged by worst directions, use:
  - `--hard-probe-iters > 0`
  - `--target-metric hard_dir`
- Keep comparing `AUTO` vs fixed `PE2` under hard-direction targets; they are not equivalent.
