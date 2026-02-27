"""Lean public API for production usage."""

__version__ = "0.2.0"

from .api import (
    PrecondConfig,
    ScheduleConfig,
    build_schedule,
    solve_gram_spd,
    solve_nonspd,
    solve_spd,
)

__all__ = [
    "__version__",
    "ScheduleConfig",
    "PrecondConfig",
    "build_schedule",
    "solve_spd",
    "solve_nonspd",
    "solve_gram_spd",
]
