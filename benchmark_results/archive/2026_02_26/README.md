# 2026-02-26 Benchmark Snapshots

This directory is organized into hierarchical tiers to distinguish verified production results from historical research.

## Tier 0: LATEST PRODUCTION (Up-to-Date)

These folders contain the final, verified results using the current $k$-aware branching logic and production-ready solvers.

- **`0_LATEST_PRODUCTION/final_report/`**
    - **[LATEST]** The definitive summary of the GEMM-heavy solver architecture.
    - Includes benchmarks for the production auto-solver (`apply_inverse_root_auto`) across all $k$ values.
    - Summary: [summary.md](file:///d:/GitHub/JiHa-Kim/fast-matrix-inverse-roots/benchmark_results/2026_02_26/0_LATEST_PRODUCTION/final_report/summary.md)

- **`0_LATEST_PRODUCTION/nonspd_full_suite/`**
    - High-fidelity production logs for general matrices in **fp32** and **bf16**.
    - Benchmarks the final branching logic vs `torch.linalg.solve`.
    - Logs: `log_fp32.txt`, `log_bf16.txt`.

- **`0_LATEST_PRODUCTION/spd_full_suite/`**
    - High-fidelity production logs for SPD matrices.
    - Logs: `log_fp32.txt`, `log_bf16.txt`.

---

## Tier 1: RESEARCH ARCHIVE (Historical/Outdated)

These folders contain the data collected during the iterative research and ablation phases. They may use outdated code or sub-optimal parameters.

- `1_RESEARCH_ARCHIVE/idea_solve_inverse_ablation_t10/`: Original solver ablation (NSRC, CG, Chebyshev).
- `1_RESEARCH_ARCHIVE/perf_coupled_affine_fastpath_t20/`: Optimization fast-path A/B tests.
- `1_RESEARCH_ARCHIVE/idea_cuda_graph_t20_warmup2/`: CUDA Graph vs eager mode research.
- `1_RESEARCH_ARCHIVE/precond/`: Ruiz and Jacobi preconditioning development.
- `1_RESEARCH_ARCHIVE/idea_affine_online_t20/`: Early greedy-affine schedule research.
