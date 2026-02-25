from .auto_policy import AutoPolicyConfig, choose_auto_method
from .coeffs import _quad_coeffs, build_pe_schedules
from .coupled import (
    IrootWorkspaceCoupled,
    IsqrtWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_sqrt_pe_quadratic,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from .precond import PrecondStats, precond_spd
from .uncoupled import (
    IrootWorkspaceUncoupled,
    inverse_proot_pe_quadratic_uncoupled,
)
from .apply import apply_inverse, apply_inverse_root
from .metrics import (
    QualityStats,
    compute_quality_stats,
    exact_inverse_proot,
    exact_inverse_sqrt,
    iroot_relative_error,
    isqrt_relative_error,
)
from .chebyshev import (
    apply_inverse_chebyshev,
    apply_inverse_proot_chebyshev,
    ChebyshevApplyWorkspace,
)

__all__ = [
    "AutoPolicyConfig",
    "choose_auto_method",
    "_quad_coeffs",
    "build_pe_schedules",
    "IrootWorkspaceCoupled",
    "IsqrtWorkspaceCoupled",
    "InverseSolveWorkspaceCoupled",
    "inverse_sqrt_pe_quadratic",
    "inverse_proot_pe_quadratic_coupled",
    "inverse_solve_pe_quadratic_coupled",
    "PrecondStats",
    "precond_spd",
    "IrootWorkspaceUncoupled",
    "inverse_proot_pe_quadratic_uncoupled",
    "QualityStats",
    "compute_quality_stats",
    "exact_inverse_proot",
    "exact_inverse_sqrt",
    "iroot_relative_error",
    "isqrt_relative_error",
    "apply_inverse",
    "apply_inverse_root",
    "apply_inverse_chebyshev",
    "apply_inverse_proot_chebyshev",
    "ChebyshevApplyWorkspace",
]
