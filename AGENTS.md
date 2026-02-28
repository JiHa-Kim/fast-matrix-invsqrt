# AGENT mandates

## A/B Testing & Integration
- **Mandatory A/B Testing**: EVERY change intended for integration into the main branch MUST be carefully A/B tested using the `run_benchmarks.py` infrastructure.
- **Baseline Selection**: Comparisons MUST be made against the **best previous version** (the current performance leader in production reports), not just the immediately preceding commit.
- **Integration Criteria**: A change is only eligible for integration if it provides a **definite win** in terms of speed (ms), quality (residual/relerr), and stability (failure rates) without non-negligible regressions.
- **Handling Regressions/Lateral Moves**: If a change does not provide a clear advantage:
    1.  **Document**: Detailed findings, including why it underperformed or was lateral, MUST be added to `docs/methods/benchmark_decisions.md`.
    2.  **Archive**: The implementation should be moved to the `archive/` directory or a research branch. Do NOT merge it into the main kernel paths.

## Benchmarking Protocol
- Use `--prod` flags for final validation to ensure consistency with documentation.
- Always report `kappa_proxy` and `steps` for iterative methods to track efficiency gains.
- Distinguish between "non-finite" and "quality" failures in all reports.
