import math

from fast_iroot.coeff_tuner import (
    affine_b_feasible_bounds,
    affine_coeffs_from_b,
    affine_qmin,
    certify_positivity_quadratic,
    coupled_apply_step_gemm_cost,
    fit_quadratic_local,
    interval_log_width,
    interval_update_affine_exact,
    interval_update_quadratic_exact,
    interval_error_to_identity,
    inverse_newton_coeffs,
    local_quadratic_coeffs_from_alpha,
    make_schedule,
    plan_coupled_quadratic_affine_opt_schedule,
    plan_coupled_local_minimax_schedule,
    plan_coupled_quadratic_newton_schedule,
    solve_local_affine_b_optimal,
    solve_local_alpha_minimax,
)


def test_certify_positivity_quadratic():
    # q(y) = 2.0 (constantly positive)
    assert certify_positivity_quadratic(2.0, 0.0, 0.0, 0.1, 1.0) is True

    # q(y) = -1.0 (constantly negative)
    assert certify_positivity_quadratic(-1.0, 0.0, 0.0, 0.1, 1.0) is False

    # q(y) = y - 0.5 (positive on [0.6, 1.0], negative on [0.1, 0.4])
    assert certify_positivity_quadratic(-0.5, 1.0, 0.0, 0.6, 1.0) is True
    assert certify_positivity_quadratic(-0.5, 1.0, 0.0, 0.1, 0.4) is False

    # q(y) = (y - 0.5)^2 + 0.1 = 0.35 - y + y^2 (vertex at 0.5, min 0.1)
    assert certify_positivity_quadratic(0.35, -1.0, 1.0, 0.1, 1.0) is True

    # q(y) = (y - 0.5)^2 - 0.1 (vertex at 0.5, min -0.1)
    # Should fail if interval contains 0.5
    assert hasattr(certify_positivity_quadratic, "__call__")
    assert certify_positivity_quadratic(-0.1 + 0.25, -1.0, 1.0, 0.1, 0.9) is False

    # But passes if we only check [0.8, 1.0] where min is at 0.8
    # q(0.8) = 0.3^2 - 0.1 = -0.01 (still negative)
    # q(0.9) = 0.4^2 - 0.1 = 0.06 > 0
    # Let's test [0.9, 1.0]
    assert certify_positivity_quadratic(-0.1 + 0.25, -1.0, 1.0, 0.9, 1.0) is True


def test_inverse_newton_coeffs():
    # p=2: a = 1.5, b = -0.5, c = 0
    a, b, c = inverse_newton_coeffs(2)
    assert math.isclose(a, 1.5)
    assert math.isclose(b, -0.5)
    assert c == 0.0


def test_affine_coeffs_from_b_fixed_point():
    for b_slope in (-2.0, -0.5, 0.0, 0.7):
        a, b, c = affine_coeffs_from_b(b_slope)
        assert math.isclose(c, 0.0)
        # q(1) = 1 for the fixed-point-preserving affine family.
        assert math.isclose(a + b, 1.0)


def test_interval_update_affine_exact_matches_quadratic_exact_for_affine():
    lo, hi = 0.2, 1.0
    for p in [1, 2, 4]:
        b_slope = -1.0 / p  # inverse-Newton affine
        abc = affine_coeffs_from_b(b_slope)
        lo1, hi1 = interval_update_affine_exact(b_slope, lo, hi, p_val=p)
        lo2, hi2 = interval_update_quadratic_exact(abc, lo, hi, p_val=p)
        assert math.isclose(lo1, lo2, rel_tol=1e-8, abs_tol=1e-10)
        assert math.isclose(hi1, hi2, rel_tol=1e-8, abs_tol=1e-10)


def test_affine_b_feasible_bounds_contains_newton():
    lo, hi = 0.2, 1.0
    b_lo, b_hi = affine_b_feasible_bounds(lo, hi, q_floor=1e-6)
    b_ns = -0.5  # p=2
    assert b_lo < b_hi
    assert b_lo < b_ns < b_hi
    assert affine_qmin(b_ns, lo, hi) > 1e-6

    # p=4: a = 1.25, b = -0.25, c = 0
    a, b, c = inverse_newton_coeffs(4)
    assert math.isclose(a, 1.25)
    assert math.isclose(b, -0.25)
    assert c == 0.0


def test_interval_update_quadratic_exact():
    # p=2, Newton step: q(y) = 1.5 - 0.5y
    # phi(y) = y * (1.5 - 0.5y)^2 = 2.25y - 1.5y^2 + 0.25y^3
    # phi'(y) = 2.25 - 3y + 0.75y^2
    # Roots of phi'(y) = 0: y^2 - 4y + 3 = 0 => y = 1, y = 3
    # phi(1) = 1 * (1)^2 = 1.0
    # phi(3) = 3 * (0)^2 = 0.0
    # on interval [0.5, 2.0]:
    # candidates: phi(0.5) = 0.5 * 1.25^2 = 0.78125
    #             phi(1.0) = 1.0
    #             phi(2.0) = 2.0 * 0.5^2 = 0.5
    # min should be 0.5, max should be 1.0
    a, b, c = 1.5, -0.5, 0.0
    lo, hi = interval_update_quadratic_exact([a, b, c], 0.5, 2.0, p_val=2)
    assert math.isclose(lo, 0.5)
    assert math.isclose(hi, 1.0)


def test_fit_quadratic_local():
    # Ensure the returned coefficients satisfy q(1)=1 and q'(1)=-1/p
    for p in [2, 3, 4]:
        abc = fit_quadratic_local(0.1, 1.0, p_val=p, iters=10)
        a, b, c = abc

        # q(1) = a + b + c
        q_1 = a + b + c
        assert math.isclose(q_1, 1.0, abs_tol=1e-5), f"q(1) != 1 for p={p}: {q_1}"

        # q'(1) = b + 2c
        qp_1 = b + 2 * c
        assert math.isclose(qp_1, -1.0 / p, abs_tol=1e-5), (
            f"q'(1) != -1/{p} for p={p}: {qp_1}"
        )


def test_make_schedule_certified_vs_legacy():
    # Smoke test to ensure make_schedule runs and intervals are monotonic
    sched_cert = make_schedule(
        kind="quad", T=3, l0=0.05, u0=1.0, certified=True, p_val=2
    )
    sched_legacy = make_schedule(
        kind="quad", T=3, l0=0.05, u0=1.0, certified=False, p_val=2
    )

    assert len(sched_cert) == 3
    assert len(sched_legacy) == 3

    # Check interval propagation in certified mode (should grow towards 1.0)
    lo0 = 0.05
    for t, step in enumerate(sched_cert):
        a, b, c, lo, hi = step
        assert lo >= lo0, f"Step {t}: lo={lo} went down from {lo0}"
        lo0 = lo


def test_coupled_apply_step_gemm_cost_affine_saves_one_gemm():
    # Affine fast path should save exactly one GEMM from B formation.
    for p in [1, 2, 3, 4, 7]:
        full = coupled_apply_step_gemm_cost(p, affine_step=False, include_y_update=True)
        aff = coupled_apply_step_gemm_cost(p, affine_step=True, include_y_update=True)
        assert full - aff == 1


def test_plan_coupled_quadratic_newton_schedule_picks_newton_on_bad_base():
    base = [(1.0, 0.0, 0.0)] * 4  # identity map (no contraction)
    sched, meta = plan_coupled_quadratic_newton_schedule(
        base, p_val=2, lo_init=0.2, hi_init=1.0
    )

    assert len(sched) == len(base)
    ns = inverse_newton_coeffs(2)
    assert any(
        math.isclose(a, ns[0]) and math.isclose(b, ns[1]) and math.isclose(c, ns[2])
        for (a, b, c) in sched
    )
    assert meta["newton_steps"] >= 1.0
    assert meta["pred_err_final"] <= interval_error_to_identity(0.2, 1.0)


def test_plan_coupled_quadratic_newton_schedule_respects_min_improve_gate():
    base = [(1.4, -0.4, 0.0)] * 3
    sched, meta = plan_coupled_quadratic_newton_schedule(
        base,
        p_val=2,
        lo_init=0.3,
        hi_init=1.0,
        min_rel_improve=1.0,  # impossible threshold; should never switch
    )
    assert sched == base
    assert math.isclose(meta["newton_steps"], 0.0)


def test_local_quadratic_coeffs_from_alpha_contains_newton():
    for p in [1, 2, 4]:
        a, b, c = local_quadratic_coeffs_from_alpha(0.0, p)
        an, bn, cn = inverse_newton_coeffs(p)
        assert math.isclose(a, an)
        assert math.isclose(b, bn)
        assert math.isclose(c, cn)


def test_solve_local_alpha_minimax_no_worse_than_newton_objective():
    lo, hi = 0.2, 1.0
    alpha, meta = solve_local_alpha_minimax(p_val=2, lo=lo, hi=hi)
    assert math.isfinite(alpha)
    assert meta["fallback_ns"] in (0.0, 1.0)

    a_mm, b_mm, c_mm = local_quadratic_coeffs_from_alpha(alpha, 2)
    a_ns, b_ns, c_ns = inverse_newton_coeffs(2)
    lo_mm, hi_mm = interval_update_quadratic_exact([a_mm, b_mm, c_mm], lo, hi, p_val=2)
    lo_ns, hi_ns = interval_update_quadratic_exact([a_ns, b_ns, c_ns], lo, hi, p_val=2)
    err_mm = interval_error_to_identity(lo_mm, hi_mm)
    err_ns = interval_error_to_identity(lo_ns, hi_ns)
    assert err_mm <= err_ns + 1e-8


def test_solve_local_affine_b_optimal_no_worse_than_newton():
    lo, hi = 0.2, 1.0
    for p in [1, 2, 4]:
        b_star, meta = solve_local_affine_b_optimal(p_val=p, lo=lo, hi=hi)
        assert math.isfinite(b_star)
        assert meta["fallback_ns"] in (0.0, 1.0)

        lo_aff, hi_aff = interval_update_affine_exact(b_star, lo, hi, p_val=p)
        b_ns = -1.0 / p
        lo_ns, hi_ns = interval_update_affine_exact(b_ns, lo, hi, p_val=p)
        err_aff = interval_error_to_identity(lo_aff, hi_aff)
        err_ns = interval_error_to_identity(lo_ns, hi_ns)
        assert err_aff <= err_ns + 1e-8


def test_plan_coupled_local_minimax_schedule_reports_step_counts():
    base = [(1.0, 0.0, 0.0)] * 3
    sched, meta = plan_coupled_local_minimax_schedule(
        base,
        p_val=2,
        lo_init=0.2,
        hi_init=1.0,
        min_rel_improve=0.0,
        min_ns_logwidth_rel_improve=0.0,
    )
    assert len(sched) == 3
    total_steps = (
        meta["base_steps"] + meta["newton_steps"] + meta["minimax_steps"]
    )
    assert math.isclose(total_steps, 3.0)
    assert math.isfinite(meta["pred_err_final"])
    assert math.isfinite(interval_log_width(meta["pred_lo_final"], meta["pred_hi_final"]))


def test_plan_coupled_quadratic_affine_opt_schedule_reports_step_counts():
    base = [(1.0, 0.0, 0.0)] * 3
    sched, meta = plan_coupled_quadratic_affine_opt_schedule(
        base,
        p_val=2,
        lo_init=0.2,
        hi_init=1.0,
        min_rel_improve=0.0,
    )
    assert len(sched) == 3
    total_steps = (
        meta["base_steps"] + meta["newton_steps"] + meta["affine_opt_steps"]
    )
    assert math.isclose(total_steps, 3.0)
    assert math.isfinite(meta["pred_err_final"])
