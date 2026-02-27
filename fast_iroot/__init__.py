from .coeffs import _quad_coeffs, build_pe_schedules
from .coupled import (
    IrootWorkspaceCoupled,
    IsqrtWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_sqrt_pe_quadratic,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from .precond import (
    DUAL_GRAM_PRECOND_MODES,
    GRAM_PRECOND_MODES,
    SPD_PRECOND_MODES,
    PrecondStats,
    precond_gram_dual_spd,
    precond_gram_spd,
    precond_spd,
)
from .uncoupled import (
    IrootWorkspaceUncoupled,
    inverse_proot_pe_quadratic_uncoupled,
)
from .apply import (
    DualGramInverseApplyWorkspace,
    GramInverseApplyWorkspace,
    InverseApplyAutoWorkspace,
    apply_inverse,
    apply_inverse_sqrt_spd,
    apply_inverse_sqrt_non_spd,
    apply_inverse_root_gram_rhs_spd,
    apply_inverse_root_gram_spd,
    apply_inverse_sqrt_gram_spd,
    apply_inverse_root,
    apply_inverse_root_auto,
)
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

from .nsrc import (
    NSRCWorkspace,
    nsrc_solve,
    nsrc_solve_preconditioned,
    hybrid_pe_nsrc_solve,
)
from .nonspd import NONSPD_PRECOND_MODES, precond_nonspd

__all__ = [
    "_quad_coeffs",
    "build_pe_schedules",
    "IrootWorkspaceCoupled",
    "IsqrtWorkspaceCoupled",
    "InverseSolveWorkspaceCoupled",
    "inverse_sqrt_pe_quadratic",
    "inverse_proot_pe_quadratic_coupled",
    "inverse_solve_pe_quadratic_coupled",
    "PrecondStats",
    "SPD_PRECOND_MODES",
    "GRAM_PRECOND_MODES",
    "DUAL_GRAM_PRECOND_MODES",
    "precond_spd",
    "precond_gram_spd",
    "precond_gram_dual_spd",
    "IrootWorkspaceUncoupled",
    "inverse_proot_pe_quadratic_uncoupled",
    "QualityStats",
    "compute_quality_stats",
    "exact_inverse_proot",
    "exact_inverse_sqrt",
    "iroot_relative_error",
    "isqrt_relative_error",
    "apply_inverse",
    "apply_inverse_sqrt_spd",
    "apply_inverse_sqrt_non_spd",
    "apply_inverse_root_gram_rhs_spd",
    "apply_inverse_root_gram_spd",
    "apply_inverse_sqrt_gram_spd",
    "apply_inverse_root",
    "apply_inverse_root_auto",
    "InverseApplyAutoWorkspace",
    "GramInverseApplyWorkspace",
    "DualGramInverseApplyWorkspace",
    "apply_inverse_chebyshev",
    "apply_inverse_proot_chebyshev",
    "ChebyshevApplyWorkspace",
    "NSRCWorkspace",
    "nsrc_solve",
    "nsrc_solve_preconditioned",
    "hybrid_pe_nsrc_solve",
    "NONSPD_PRECOND_MODES",
    "precond_nonspd",
]
