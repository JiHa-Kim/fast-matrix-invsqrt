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

- `idea_cuda_graph_t20_warmup2/`
  - Coupled-apply CUDA graph ablation (`--cuda-graph` off vs on).
  - Corrected timing method uses `--timing-warmup-reps 2` to reduce first-run bias.
  - Raw solve logs: `off/`, `on/`.
  - Aggregated summary: `summary_coupled_apply.md`.
  - Balanced paired validation: `paired_balanced_primed.md`.
  - Profile evidence: `profile/off_vs_on_profile.txt`, `profile/summary.md`.

- `nonspd_p1_suite/`
  - Dedicated non-SPD solve suite (`p=1`) at `n=1024`, `k in {1,16,64}`, `10` trials.
  - Legacy canonical logs: `t10_fp32/`.
  - Early-safe logs (with `--nonspd-safe-early-y-tol 0.8`): `t10_fp32_safe_early/`.
  - Cross-size sanity (`n in {256,512,1024}`, `k=16`): `cross_size_k16_t10_fp32_safe_early/`.
  - Summaries: `summary_t10_fp32.md`, `summary_t10_fp32_safe_early.md`.
  - Exploratory runs: `exploratory/`.

- `idea_solve_inverse_ablation_t10/`
  - One-factor-at-a-time non-SPD `p=1` ideas ablation.
  - Canonical summary (`k=16`): `summary.md`.
  - Per-RHS summaries: `summary_k1.md`, `summary_k16.md`, `summary_k64.md`.
  - Runbook: `README.md`.

- `spd_p1_suite/`
  - SPD solve suite (`p=1`) with exact baselines:
    - `Torch-Solve`
    - `Torch-Cholesky-Solve`
  - `n=1024`, `k in {1,16,64}`, `10` trials:
    - `t10_fp32_cholesky/`
  - Cross-size sanity (`n in {256,512,1024}`, `k=16`):
    - `cross_size_k16_t10_fp32_cholesky/`
  - Comprehensive case sweep (`gaussian_spd`, `illcond_1e6`, `illcond_1e12`, `near_rank_def`, `spike`):
    - `comprehensive_k16_t10_fp32_cholesky/`
  - Summaries:
    - `summary_t10_fp32_cholesky.md`
    - `summary_comprehensive_k16_t10_fp32_cholesky.md`
