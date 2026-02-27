# A/B Summary: `p=2` Kappa-Table Candidate (k<n-only wiring)

Date: 2026-02-26

Decision:
- Reject candidate globally (paired with `p=4` rejection below).

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode kappa-minimax-table-klt-only" --ab-label-a greedy_affine_opt --ab-label-b kappa_table_klt_only --ab-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p2/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- Total across 12 cells: `-1.20%` total ms.
- `k<n`: `-4.75%`.
- `k=n`: `+2.55%`.
- Relerr drift: negligible (`max ~0.0%` relative drift in this run).

Conclusion:
- `p=2` gains are modest and not enough to justify adoption when paired with `p=4` behavior.
