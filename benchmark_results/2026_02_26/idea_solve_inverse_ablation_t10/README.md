# Solve-Inverse Ideas Ablation (`t10`)

One-factor-at-a-time non-SPD `p=1` ablations derived from `ideas/solve_inverse.md`.

## Scope

- script: `scripts/ablate_solve_inverse_ideas.py`
- size: `n=1024`
- RHS widths: `k in {1,16,64}`
- trials: `10`
- timing: `--timing-reps 5 --timing-warmup-reps 2`
- dtype: `fp32`
- cases: `gaussian_shifted,nonnormal_upper,similarity_posspec,similarity_posspec_hard`

## Files

- `summary.md`: canonical summary (`k=16`)
- `summary_k1.md`
- `summary_k16.md`
- `summary_k64.md`

## Ablation Cells

- `A0`: baseline (`row-norm`, tuned coefficients, coupled apply)
- `A1`: precondition only (`frob`)
- `A2`: precondition only (`ruiz`)
- `A3`: coefficient only (`tuned` with safety multiplier `x1.2`)
- `A4`: final-residual safe fallback only
- `A5`: final-residual + early safe fallback
- `A6`: adaptive runtime switching + safety
- `A7`: exact reference (`Torch-Solve`)

## Reproduce

```bash
uv run python scripts/ablate_solve_inverse_ideas.py --sizes 1024 --k 1  --trials 10 --dtype fp32 --timing-reps 5 --timing-warmup-reps 2 --out-md benchmark_results/2026_02_26/idea_solve_inverse_ablation_t10/summary_k1.md
uv run python scripts/ablate_solve_inverse_ideas.py --sizes 1024 --k 16 --trials 10 --dtype fp32 --timing-reps 5 --timing-warmup-reps 2 --out-md benchmark_results/2026_02_26/idea_solve_inverse_ablation_t10/summary_k16.md
uv run python scripts/ablate_solve_inverse_ideas.py --sizes 1024 --k 64 --trials 10 --dtype fp32 --timing-reps 5 --timing-warmup-reps 2 --out-md benchmark_results/2026_02_26/idea_solve_inverse_ablation_t10/summary_k64.md
```

Run sequentially (one process at a time) to avoid timing interference.
