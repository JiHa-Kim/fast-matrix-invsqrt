# A/B Summary: `p=4` Kappa-Table Candidate (k<n-only wiring)

Date: 2026-02-26

Decision:
- Reject candidate (`kappa-minimax-table-klt-only`).

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode kappa-minimax-table-klt-only" --ab-label-a greedy_affine_opt --ab-label-b kappa_table_klt_only --ab-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p4/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- Total across 12 cells: `+17.29%` total ms (regression).
- `k<n`: `+28.74%` (regression).
- `k=n`: `+7.80%` (regression).
- Relerr drift: small (`max ~3.51%` relative drift), but speed regression is dominant.

Conclusion:
- Candidate is not viable even with k<n-only wiring in this matrix.
