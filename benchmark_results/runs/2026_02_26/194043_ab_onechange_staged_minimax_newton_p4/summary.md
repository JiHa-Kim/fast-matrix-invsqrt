# A/B Summary: `p=4` Staged Minimax+Newton Tail (candidate)

Date: 2026-02-26

Decision:
- Reject candidate (`staged-minimax-newton`, tail=2).

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode staged-minimax-newton --online-coeff-staged-newton-tail 2" --ab-label-a greedy_affine_opt --ab-label-b staged_minimax_newton --ab-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p4/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- Total across 12 cells: `+7.66%` total ms (regression).
- `k<n`: `+9.09%`.
- `k=n`: `+5.99%`.
- Win/loss count: `4` faster, `8` slower.
- Relerr drift: up to `~14.8%` relative increase on some cells.

Conclusion:
- Keep current default `greedy-affine-opt`; do not add staged mode.
