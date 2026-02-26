# Benchmark Results (2026-02-25)

## Structure

- `solve_ablation_t20/`
  - 20-trial solve ablation logs for online coefficient modes:
    - `results_solve_off_t20.txt`
    - `results_solve_greedy_newton_t20.txt`
    - `results_solve_greedy_minimax_t20.txt`
    - `results_solve_auto_t20.txt` (older auto mapping run)
    - `results_solve_auto_newdefault_t20.txt` (post-default-change run)
- `solve_exploratory/`
  - exploratory solve runs used while iterating on online-stop and schedule ideas.
- `idea3_square_rhs_t20/`
  - square-RHS validation (`k=1024`) for direct coupled apply vs materialize-then-multiply.
- `iroot_p1_p5/`
  - refreshed inverse-root sweep logs (`results_p1.txt` to `results_p5.txt`).

## Main Report

- Summary and winner selection are documented in:
  - `reports/2026_02_25_solve_online_coeff_ablation_t20.md`
- Direct-vs-materialize square-RHS validation is documented in:
  - `reports/2026_02_25_idea3_square_rhs_apply_vs_materialize.md`
