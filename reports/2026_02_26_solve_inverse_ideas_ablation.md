# Solve-Inverse Ideas Ablation (`p=1`, non-SPD)

*Date: 2026-02-26*

## Goal

Evaluate ideas from `ideas/solve_inverse.md` using controlled ablations, changing one factor at a time from a fixed baseline.

Baseline (`A0`):

- `PE-Quad-Coupled-Apply`
- preconditioner: `row-norm`
- coefficients: tuned safety `x1.00`

Sweep setup:

- `n=1024`, `k in {1,16,64}`, `trials=10`, `fp32`
- cases: `gaussian_shifted`, `nonnormal_upper`, `similarity_posspec`, `similarity_posspec_hard`
- artifacts: `benchmark_results/2026_02_26/idea_solve_inverse_ablation_t10/`

## Key Findings

1. Preconditioning ideas:
- `A1` (`frob`) is unsafe: catastrophic error on `similarity_posspec` (`~4.5e10` relerr) despite similar speed.
- `A2` (`ruiz`) is consistently slower than baseline and does not fix hard-case divergence.
- Decision: keep `row-norm` for non-SPD `p=1`.

2. Coefficient schedule safety:
- `A3` (`tuned` safety `x1.2`) causes large moderate-case error (`~2e-1`) with no hard-case recovery.
- Decision: keep tuned safety `x1.0`; conservative scaling is harmful here.

3. Safety policy ablation:
- `A4` (final fallback only) removes hard-case failure but pays large hard-case latency.
- `A5` (final + early fallback) preserves hard-case robustness and is materially faster than `A4` on `similarity_posspec_hard`:
  - `k=1`: `5.264 ms` vs `7.040 ms`
  - `k=16`: `5.561 ms` vs `7.367 ms`
  - `k=64`: `5.595 ms` vs `7.443 ms`
- `A6` (adaptive + safety) is slower than `A5` in these runs and adds moderate-case overhead.

## Practical Recommendation

- Throughput-first moderate workloads: keep `A0` path (`PE-Quad-Coupled-Apply`, `row-norm`).
- Robustness-first unknown workloads: use `A5` (`PE-Quad-Coupled-Apply-Safe` + early guard).
- Do not use `frob`, `ruiz`, or conservative coefficient safety `x1.2` for this non-SPD `p=1` setting.
