# AGENT mandates

## A/B Testing & Integration
- **Mandatory A/B Testing**: EVERY change intended for integration into the main branch MUST be carefully A/B tested using the `run_benchmarks.py` infrastructure.
- **Baseline Selection**: Comparisons MUST be made against the **official baseline** (stored in `benchmark_results/baseline/baseline_solver.json`), which represents the current performance leader. Use the `--baseline` flag (enabled by default).
- **Manual Baseline Promotion**: The baseline is NEVER updated automatically. If a change achieves a definitive performance gain, it MUST be manually promoted using the `--update-baseline` flag.
- **Integration Criteria**: A change is only eligible for integration if it provides a **definite win** in terms of speed (ms), quality (residual/relerr), and stability (failure rates) without non-negligible regressions.
- **Reporting Hierarchy**: Reports are grouped by `Kind -> p -> Size -> Case`. Focus on high-fidelity 16-column metrics (residuals, median vs tail error, etc.). **Bold values** indicate the best performer for that specific metric in a scenario.

## Benchmarking Protocol
- **Publication Workflow**: Use `--prod --from-baseline` to update the production documentation once a new baseline is promoted.
- **Granular Filtering**: Use `--kinds`, `--sizes`, `--p-vals`, and `--methods` to run targeted benchmark subsets during development.
- **Integrity**: Always ensure `--integrity-checksums` are updated when changing production reports.
