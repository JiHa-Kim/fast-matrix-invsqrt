# Non-SPD Solve Suite (`p=1`, `t10`, `n=1024`)

*Date: 2026-02-26*

## Scope

Dedicated non-SPD inverse-solve benchmark using:

- script: `scripts/matrix_solve_nonspd.py`
- matrix size: `1024 x 1024`
- RHS widths: `k in {1,16,64}`
- cases: `gaussian_shifted`, `nonnormal_upper`, `similarity_posspec`, `similarity_posspec_hard`
- `10` trials per cell, `5` timing reps, `2` timing warmup reps
- dtype: `fp32`

Artifacts:

- raw logs: `benchmark_results/2026_02_26/nonspd_p1_suite/t10_fp32/`
- parsed summary: `benchmark_results/2026_02_26/nonspd_p1_suite/summary_t10_fp32.md`

## Results

For the first three cases (`gaussian_shifted`, `nonnormal_upper`, `similarity_posspec`):

- `PE-Quad-Coupled-Apply` is fastest at roughly `2.65` to `2.84 ms` mean.
- Relative error against `torch.linalg.solve` is around `5e-4` to `8e-4`.

For the hard non-normal case (`similarity_posspec_hard`):

- Fast PE methods (`PE-Quad-Coupled-Apply`, `PE-Quad-Inverse-Multiply`) are fast but fail (`~9.988e-01` relerr).
- `PE-Quad-Coupled-Apply-Adaptive` with safe fallback (`--nonspd-safe-fallback-tol 0.01`) recovers robust output (`0.0` relerr in this sweep) at much higher runtime (`~17.2 ms` mean).
- `Torch-Solve` remains stable (`~6.47e-03` mean relerr here) at `~4.57 ms` mean.

## Practical Policy

- Throughput-first moderate non-SPD workloads:
  - use `PE-Quad-Coupled-Apply` (`assume_spd=False` path).
- Robustness-first / unknown non-normality:
  - use adaptive safe mode (`nonspd_adaptive=True`, `nonspd_safe_fallback_tol=0.01`) to cap catastrophic failures.

