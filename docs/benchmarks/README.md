# Benchmark Reports

- **[Production Solver Benchmark](benchmark_results_production.md)**: High-fidelity solver suite report grouped by hierarchy (`Kind -> p -> Size -> Case`).
- **[Spectral Convergence Benchmark](spectral_convergence_production.md)**: Worst-case PE-Quad vs Newton-Schulz spectral contraction analysis.

## Development Workflow

The benchmarking system uses an **Official Baseline** stored in `benchmark_results/baseline/` to track performance improvements.

### 1. Run & Compare
Run a subset of benchmarks and automatically compare against the baseline:
```bash
# Compare SPD p=2 cases for size 1024
uv run python -m benchmarks.run_benchmarks --kinds spd --p-vals 2 --sizes 1024
```

### 2. Manual Promotion
If your changes improve performance, manually update the baseline:
```bash
uv run python -m benchmarks.run_benchmarks --kinds spd --update-baseline
```

### 3. Publish to Production
Update the official production documentation from your curated baseline:
```bash
uv run python -m benchmarks.run_benchmarks --prod --from-baseline
```

## Reproduce Full Suite

```bash
# Generate full suite report
uv run python -m benchmarks.run_benchmarks --run-name production_fullsuite

# Generate spectral convergence report
uv run python -m benchmarks.spectral_convergence --run-name spectral_convergence
```

The system writes manifests, reproducibility fingerprints, and integrity `.sha256` sidecars by default.
