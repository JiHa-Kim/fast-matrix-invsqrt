# Repository Guidelines

## Project Structure & Module Organization
- `fast_iroot/`: core library code (inverse p-th-root kernels, solve/apply paths, preconditioning, diagnostics).
- `benchmarks/`: benchmark CLIs (`run_benchmarks.py`, `solve/matrix_solve.py`, `solve/matrix_solve_nonspd.py`).
- `tests/`: `pytest` test suite (`test_*.py`) for kernels, benchmark helpers, preconditioners, and coefficient tuning.
- `benchmark_results/`: raw benchmark logs and summaries (dated folders).
- `reports/`: narrative benchmark writeups and decisions.
- `docs/`, `ideas/`, `archive/`: method notes, experimental plans, and historical artifacts.

## Build, Test, and Development Commands
- `uv sync`: install runtime and dev dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python -m pytest -q`: run all tests.
- `uv run python -m pytest tests/test_fast_iroot_fixes.py -q`: run focused regression tests.
- `uv run python -m ruff check .`: lint Python sources.
- `uv run python -m pytest tests/test_verify_iroot.py -q`: correctness/stability validation sweep.
- `uv run python benchmarks/run_benchmarks.py`: maintained full benchmark driver (rigorous defaults: trials/reps/warmups).
- `uv run python benchmarks/run_benchmarks.py --markdown --out benchmark_results/latest_solver_benchmarks.md`: regenerate the consolidated benchmark markdown report.
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n" --ab-extra-args-a "--online-coeff-target-interval-err 0.0" --ab-extra-args-b "--online-coeff-target-interval-err 0.01" --ab-out benchmark_results/latest_solver_benchmarks_ab.md`: focused A/B compare on a filtered subset (avoids running the entire matrix).
- `uv run python benchmarks/solve/matrix_solve.py --p 1 --sizes 1024 --k 16 --trials 10 --dtype fp32`: SPD solve benchmark.
- `uv run python benchmarks/solve/matrix_solve_nonspd.py --p 1 --sizes 1024 --k 16 --trials 10 --dtype fp32`: non-SPD solve benchmark.

## Coding Style & Naming Conventions
- Language: Python; use 4-space indentation and keep functions small and explicit.
- Naming: `snake_case` for functions/variables, `PascalCase` for dataclasses, `UPPER_SNAKE_CASE` for constants.
- Prefer explicit typing (`Optional[...]`, `Tuple[...]`) and defensive input validation for public APIs.
- Run `ruff` before committing; keep new code consistent with existing module patterns.

## Testing Guidelines
- Framework: `pytest`.
- Test files: `tests/test_*.py`; test names start with `test_`.
- Add unit tests for new behavior and regression tests for numerical/stability fixes.
- For performance-sensitive changes, attach benchmark logs under `benchmark_results/YYYY_MM_DD/...` and summarize in `reports/`.
- For rigorous benchmark comparisons, keep default benchmark repetitions unless explicitly doing exploratory smoke runs.

## Commit & Pull Request Guidelines
- Follow Conventional Commits with scope when useful: `feat(solve): ...`, `perf(coupled): ...`, `bench(nonspd): ...`, `docs(...): ...`, `chore(...): ...`.
- Keep commits atomic (code/test changes separate from benchmark artifact/report updates when practical).
- PRs should include:
  - clear problem statement and behavioral impact,
  - exact commands used for tests/benchmarks,
  - links to updated logs/reports (and key numbers, not just files).


