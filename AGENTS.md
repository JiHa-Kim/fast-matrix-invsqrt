# Repository Guidelines

## Project Structure & Module Organization
- `fast_iroot/`: core library code (inverse p-th-root kernels, solve/apply paths, preconditioning, diagnostics).
- `scripts/`: benchmark and verification CLIs (`matrix_iroot.py`, `matrix_solve.py`, `matrix_solve_nonspd.py`, `verify_iroot.py`).
- `tests/`: `pytest` test suite (`test_*.py`) for kernels, benchmark helpers, preconditioners, and coefficient tuning.
- `benchmark_results/`: raw benchmark logs and summaries (dated folders).
- `reports/`: narrative benchmark writeups and decisions.
- `docs/`, `ideas/`, `archive/`: method notes, experimental plans, and historical artifacts.

## Build, Test, and Development Commands
- `uv sync`: install runtime and dev dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python -m pytest -q`: run all tests.
- `uv run python -m pytest tests/test_fast_iroot_fixes.py -q`: run focused regression tests.
- `uv run python -m ruff check .`: lint Python sources.
- `uv run python scripts/verify_iroot.py`: correctness/stability validation sweep.
- `uv run python scripts/matrix_solve.py --p 1 --sizes 1024 --k 16 --trials 10 --dtype fp32`: SPD solve benchmark.
- `uv run python scripts/matrix_solve_nonspd.py --p 1 --sizes 1024 --k 16 --trials 10 --dtype fp32`: non-SPD solve benchmark.

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

## Commit & Pull Request Guidelines
- Follow Conventional Commits with scope when useful: `feat(solve): ...`, `perf(coupled): ...`, `bench(nonspd): ...`, `docs(...): ...`, `chore(...): ...`.
- Keep commits atomic (code/test changes separate from benchmark artifact/report updates when practical).
- PRs should include:
  - clear problem statement and behavioral impact,
  - exact commands used for tests/benchmarks,
  - links to updated logs/reports (and key numbers, not just files).
