# Non-SPD Solve `p=1` Suite (2026-02-26)

Dedicated benchmark artifacts for non-SPD inverse solve:

- target: `Z = A^{-1} B`
- script: `scripts/matrix_solve_nonspd.py`
- matrix size: `n=1024`
- RHS widths: `k in {1,16,64}`
- trials: `10` per cell
- timing: `--timing-reps 5 --timing-warmup-reps 2`
- dtype: `fp32`

## Layout

- `t10_fp32/`
  - canonical logs from the 10-trial sweep:
    - `solve_nonspd_p1_k1_t10_fp32.txt`
    - `solve_nonspd_p1_k16_t10_fp32.txt`
    - `solve_nonspd_p1_k64_t10_fp32.txt`
- `exploratory/`
  - earlier / exploratory runs (`t5`, duplicate filenames, and failed bf16 attempt).

## Reproduce

```bash
uv run python scripts/matrix_solve_nonspd.py --p 1 --sizes 1024 --k 1  --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/nonspd_p1_suite/t10_fp32/solve_nonspd_p1_k1_t10_fp32.txt
uv run python scripts/matrix_solve_nonspd.py --p 1 --sizes 1024 --k 16 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/nonspd_p1_suite/t10_fp32/solve_nonspd_p1_k16_t10_fp32.txt
uv run python scripts/matrix_solve_nonspd.py --p 1 --sizes 1024 --k 64 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/nonspd_p1_suite/t10_fp32/solve_nonspd_p1_k64_t10_fp32.txt
```

Note: the `Torch-Solve` baseline in this suite currently requires `fp32` on CUDA in this environment (`bf16` solve is not implemented by `torch.linalg.solve` here).

