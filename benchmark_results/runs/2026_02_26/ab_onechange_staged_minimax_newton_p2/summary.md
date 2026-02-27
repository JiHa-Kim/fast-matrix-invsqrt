# A/B Summary: `p=2` Staged Minimax+Newton Tail (candidate)

Date: 2026-02-26

Decision:
- Candidate was not adopted globally because `p=4` regressed; `p=2` result recorded here.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode staged-minimax-newton --online-coeff-staged-newton-tail 2" --ab-label-a greedy_affine_opt --ab-label-b staged_minimax_newton --ab-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p2/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- Total across 12 cells: `-12.04%` total ms.
- `k<n`: `-14.89%`.
- `k=n`: `-8.56%`.
- Win/loss count: `9` faster, `3` slower.
- Relerr drift: up to `~18%` relative increase on some cells.
