# 2026-02-26 Benchmark Artifacts

## Folders

- `idea4_precond_t20/`
  - Solve preconditioner ablation logs (`p in {1,2,4}`, `k in {1,16,64}`, 20 trials).
  - Aggregated summary: `summary_coupled_apply.md`.

- `idea4_precond_iroot_t20/`
  - IRoot preconditioner ablation logs (`p in {1,2,4}`, 20 trials).
  - Aggregated summary: `summary_pe_quad_coupled.md`.

- `idea4_gram_precond_t20/`
  - Gram-path parity/timing check for `precond_gram_spd`.
  - Summary: `summary.md`.

- `idea_affine_online_t20/`
  - Solve online-schedule ablation logs including `greedy-affine-opt`.
  - Aggregated summary: `summary_coupled_apply.md`.

- `perf_coupled_affine_fastpath_t20/`
  - A/B perf artifacts for coupled affine fast-path refactor (`HEAD` vs optimized).
  - Same-interpreter solve-suite logs: `baseline_sameenv/`, `optimized_sameenv/`.
  - Parsed suite summary: `summary_coupled_apply_fastpath_sameenv.md`.
  - Kernel-focused microbench summary: `microbench_compare.md`.
  - Operator profiler captures: `profile/baseline_top_ops.txt`, `profile/optimized_top_ops.txt`.
