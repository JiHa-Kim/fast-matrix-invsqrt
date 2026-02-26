# SPD Solve `p=1` Suite (2026-02-26)

Dedicated SPD inverse-solve benchmark artifacts with exact baselines:

- `Torch-Solve`
- `Torch-Cholesky-Solve`

Script:

- `scripts/matrix_solve.py --p 1`

Base setup:

- matrix size: `n=1024`
- RHS widths: `k in {1,16,64}`
- trials: `10` per cell
- timing: `--timing-reps 5 --timing-warmup-reps 2`
- dtype: `fp32`

## Layout

- `t10_fp32_cholesky/`
  - canonical 10-trial logs:
    - `solve_spd_p1_k1_t10_fp32_cholesky.txt`
    - `solve_spd_p1_k16_t10_fp32_cholesky.txt`
    - `solve_spd_p1_k64_t10_fp32_cholesky.txt`
- `cross_size_k16_t10_fp32_cholesky/`
  - cross-size sanity (`n in {256,512,1024}`, `k=16`):
    - `solve_spd_p1_sizes256_512_1024_k16_t10_fp32_cholesky.txt`
- `comprehensive_k16_t10_fp32_cholesky/`
  - broader SPD case sweep (`n=1024`, `k=16`):
    - `gaussian_spd,illcond_1e6,illcond_1e12,near_rank_def,spike`
    - `solve_spd_p1_n1024_k16_t10_all_cases_fp32_cholesky.txt`

## Reproduce

```bash
uv run python scripts/matrix_solve.py --p 1 --sizes 1024 --k 1  --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/spd_p1_suite/t10_fp32_cholesky/solve_spd_p1_k1_t10_fp32_cholesky.txt
uv run python scripts/matrix_solve.py --p 1 --sizes 1024 --k 16 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/spd_p1_suite/t10_fp32_cholesky/solve_spd_p1_k16_t10_fp32_cholesky.txt
uv run python scripts/matrix_solve.py --p 1 --sizes 1024 --k 64 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/spd_p1_suite/t10_fp32_cholesky/solve_spd_p1_k64_t10_fp32_cholesky.txt

uv run python scripts/matrix_solve.py --p 1 --sizes 256,512,1024 --k 16 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 > benchmark_results/2026_02_26/spd_p1_suite/cross_size_k16_t10_fp32_cholesky/solve_spd_p1_sizes256_512_1024_k16_t10_fp32_cholesky.txt

uv run python scripts/matrix_solve.py --p 1 --sizes 1024 --k 16 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype fp32 --cases gaussian_spd,illcond_1e6,illcond_1e12,near_rank_def,spike > benchmark_results/2026_02_26/spd_p1_suite/comprehensive_k16_t10_fp32_cholesky/solve_spd_p1_n1024_k16_t10_all_cases_fp32_cholesky.txt
```
