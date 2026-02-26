import math

from fast_iroot.coeff_tuner import (
    certify_positivity_quadratic,
    fit_quadratic_local,
    interval_update_quadratic_exact,
    inverse_newton_coeffs,
    make_schedule,
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
