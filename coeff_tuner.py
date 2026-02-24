import math
import warnings
import numpy as np
import torch


def smooth_max(x, tau=80.0):
    m = x.max()
    return (torch.log(torch.mean(torch.exp(tau * (x - m)))) / tau) + m


def build_grid(lo, hi, n=8192):
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 1.0001)
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n, dtype=torch.float64))


def fit_affine(
    lo,
    hi,
    init=None,
    q_floor=1e-3,
    pos_penalty=5e5,
    tau=80.0,
    iters=200,
    restarts=6,
    seed=0,
    p_val=2,
):
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)

    if init is None:
        # LS fit to y^(-1/p)
        y_np = ys.cpu().numpy()
        V = np.stack([np.ones_like(y_np), y_np], axis=1)
        target = y_np ** (-1.0 / p_val)
        init = np.linalg.lstsq(V, target, rcond=None)[0]

    base = torch.tensor(init, dtype=torch.float64)
    best, best_val = None, None

    for r in range(restarts):
        if r == 0:
            p0 = base.clone()
        else:
            noise = torch.tensor(rng.standard_normal(2), dtype=torch.float64)
            p0 = base * (1 + 0.25 * noise) + 0.1 * noise
        param = torch.nn.Parameter(p0)

        opt = torch.optim.LBFGS(
            [param], lr=0.8, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            a, b = param[0], param[1]
            q = a + b * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau) + pos_penalty * torch.mean(
                torch.relu(q_floor - q) ** 2
            )
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            a, b = param[0], param[1]
            q = a + b * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            val = err.max() + pos_penalty * torch.mean(torch.relu(q_floor - q) ** 2)
            if best_val is None or float(val) < float(best_val):
                best_val = val.clone()
                best = param.detach().clone()

    return [float(best[0]), float(best[1])]


def fit_quadratic(
    lo,
    hi,
    init=None,
    q_floor=1e-3,
    pos_penalty=5e5,
    tau=80.0,
    iters=250,
    restarts=6,
    seed=0,
    p_val=2,
):
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)

    if init is None:
        y_np = ys.cpu().numpy()
        V = np.stack([np.ones_like(y_np), y_np, y_np * y_np], axis=1)
        target = y_np ** (-1.0 / p_val)
        init = np.linalg.lstsq(V, target, rcond=None)[0]

    base = torch.tensor(init, dtype=torch.float64)
    best, best_val = None, None

    for r in range(restarts):
        if r == 0:
            p0 = base.clone()
        else:
            noise = torch.tensor(rng.standard_normal(3), dtype=torch.float64)
            p0 = base * (1 + 0.25 * noise) + 0.1 * noise
        param = torch.nn.Parameter(p0)

        opt = torch.optim.LBFGS(
            [param], lr=0.8, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            a, b, c = param[0], param[1], param[2]
            q = a + b * ys + c * ys * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau) + pos_penalty * torch.mean(
                torch.relu(q_floor - q) ** 2
            )
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            a, b, c = param[0], param[1], param[2]
            q = a + b * ys + c * ys * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            val = err.max() + pos_penalty * torch.mean(torch.relu(q_floor - q) ** 2)
            if best_val is None or float(val) < float(best_val):
                best_val = val.clone()
                best = param.detach().clone()

    return [float(best[0]), float(best[1]), float(best[2])]


# ---------------------------------------------------------------------------
# Certified machinery
# ---------------------------------------------------------------------------


def certify_positivity_quadratic(a, b, c, lo, hi, q_min=1e-6):
    """Exact positivity certification for q(y) = a + b*y + c*y^2 on [lo, hi].

    Returns True iff min(q(y)) > q_min on the interval.
    """
    q_lo = a + b * lo + c * lo * lo
    q_hi = a + b * hi + c * hi * hi
    min_val = min(q_lo, q_hi)
    # check vertex if parabola opens up and vertex is in [lo, hi]
    if c > 0:
        y_star = -b / (2.0 * c)
        if lo <= y_star <= hi:
            q_star = a + b * y_star + c * y_star * y_star
            min_val = min(min_val, q_star)
    return min_val > q_min


def inverse_newton_coeffs(p_val):
    """Return (a, b, c) for the inverse-Newton step q(y ) = (p+1-y)/p.

    In standard basis q(y) = a + b*y + c*y^2:
        a = (p+1)/p, b = -1/p, c = 0.
    """
    return (float(p_val + 1) / p_val, -1.0 / p_val, 0.0)


def _phi_and_dphi_coeffs_p2(a, b, c):
    """Return polynomial coefficients for φ(y) = y·q(y)^2 and φ'(y) when p=2.

    q(y) = a + b*y + c*y^2
    φ(y) = y·(a + b*y + c*y^2)^2  — degree-5 polynomial.
    φ'(y) = d/dy φ(y) — degree-4 polynomial.

    Returns (phi_coeffs, dphi_coeffs) as arrays in *ascending* power order,
    i.e. phi_coeffs[k] is the coefficient of y^k.
    """
    # q(y) = a + b*y + c*y^2
    # q^2 = a^2 + 2ab*y + (b^2+2ac)*y^2 + 2bc*y^3 + c^2*y^4
    # phi = y * q^2 = a^2*y + 2ab*y^2 + (b^2+2ac)*y^3 + 2bc*y^4 + c^2*y^5
    phi = np.array(
        [
            0.0,
            a * a,
            2.0 * a * b,
            b * b + 2.0 * a * c,
            2.0 * b * c,
            c * c,
        ]
    )
    # d/dy: multiply coefficient k by k
    dphi = np.array([phi[k] * k for k in range(1, 6)])
    return phi, dphi


def interval_update_quadratic_exact(abc, lo, hi, p_val=2):
    """Certified interval update via critical-point extrema.

    For p=2, finds exact roots of φ'(y) and evaluates φ at endpoints + critical
    points.  For other p, falls back to dense grid sampling with conservative
    padding.

    Returns (lo_new, hi_new).
    """
    a, b, c = abc
    lo, hi = float(lo), float(hi)

    if p_val == 2:
        phi, dphi = _phi_and_dphi_coeffs_p2(a, b, c)
        # np.roots expects *descending* power order
        roots = np.roots(dphi[::-1])
        # filter real roots inside [lo, hi]
        candidates = [lo, hi]
        for r in roots:
            if np.isreal(r):
                yr = float(np.real(r))
                if lo <= yr <= hi:
                    candidates.append(yr)
        # evaluate phi at candidates
        vals = [np.polyval(phi[::-1], y) for y in candidates]
        return float(min(vals)), float(max(vals))
    else:
        # fallback: dense grid + conservative padding
        return _interval_update_grid(abc, lo, hi, p_val=p_val)


def _interval_update_grid(abc, lo, hi, n=32768, p_val=2):
    """Grid-based interval update with conservative padding (legacy / generic p)."""
    a, b, c = abc
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys + c * ys * ys
    phi = ys * (q**p_val)
    lo_new, hi_new = float(phi.min()), float(phi.max())
    # conservative padding for non-exact case
    span = max(hi_new - lo_new, 1e-12)
    return lo_new - 0.001 * span, hi_new + 0.001 * span


def interval_update_affine(ab, lo, hi, n=16384, p_val=2):
    a, b = ab
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


def interval_update_quadratic(abc, lo, hi, n=16384, p_val=2):
    """Legacy grid-based interval update (kept for backward compatibility)."""
    a, b, c = abc
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys + c * ys * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


# ---------------------------------------------------------------------------
# Local-basis quadratic fit: q(y) = 1 - (1/p)(y-1) + α(y-1)^2
# ---------------------------------------------------------------------------


def fit_quadratic_local(
    lo,
    hi,
    tau=80.0,
    iters=300,
    restarts=8,
    seed=0,
    p_val=2,
):
    """Fit q(y) with q(1)=1 and q'(1)=-1/p baked in.

    Parameterization: q(y) = 1 - (1/p)(y-1) + α*(y-1)^2.
    Only α is free.  Returns (a, b, c) in standard basis q(y)=a+b*y+c*y^2.
    """
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)
    inv_p = 1.0 / p_val

    # To strictly guarantee q(y) > q_min on [lo, hi]:
    # q(y) = 1 - (1/p)(y-1) + alpha(y-1)^2 > q_min
    # => alpha > (q_min - (1 - (1/p)(y-1))) / (y-1)^2  for y != 1
    # We only need to check the endpoints lo and hi because the bounding curve
    # is monotonic on (0, 1) and (1, infinity).
    q_min = 1e-4
    alpha_lb_lo = -1e9
    if abs(lo - 1.0) > 1e-9:
        alpha_lb_lo = (q_min - (1.0 - inv_p * (lo - 1.0))) / ((lo - 1.0) ** 2)

    alpha_lb_hi = -1e9
    if abs(hi - 1.0) > 1e-9:
        alpha_lb_hi = (q_min - (1.0 - inv_p * (hi - 1.0))) / ((hi - 1.0) ** 2)

    alpha_min = max(alpha_lb_lo, alpha_lb_hi)

    best_alpha, best_val = None, None

    for r in range(restarts):
        if r == 0:
            a0 = (p_val + 1.0) / (2.0 * p_val * p_val)  # from φ''(1)=0 condition
        else:
            a0 = (p_val + 1.0) / (2.0 * p_val * p_val) + 0.3 * rng.standard_normal()

        a0 = max(a0, alpha_min + 1e-5)
        param = torch.nn.Parameter(torch.tensor([a0], dtype=torch.float64))

        opt = torch.optim.LBFGS(
            [param], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            # Project alpha to feasible region strictly
            alpha = torch.clamp(param[0], min=alpha_min + 1e-6)
            dy = ys - 1.0
            q = 1.0 - inv_p * dy + alpha * dy * dy
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            alpha = float(torch.clamp(param[0], min=alpha_min + 1e-6))
            dy = ys - 1.0
            q = 1.0 - inv_p * dy + alpha * dy * dy
            err = torch.abs(1.0 - ys * (q**p_val))
            val = float(err.max())
            if best_val is None or val < best_val:
                best_val = val
                best_alpha = alpha

    # Convert local basis to standard basis: q(y) = a + b*y + c*y^2
    # q(y) = 1 - (1/p)(y-1) + α(y-1)^2
    #       = (1 + 1/p + α) + (-1/p - 2α)*y + α*y^2
    a_std = 1.0 + inv_p + best_alpha
    b_std = -inv_p - 2.0 * best_alpha
    c_std = best_alpha
    return [a_std, b_std, c_std]


def make_schedule(
    kind="affine",
    T=3,
    l0=0.05,
    u0=1.0,
    l_cushion=0.05,
    safety=1.0,
    seed=0,
    p_val=2,
    certified=True,
):
    """Build a polynomial coefficient schedule.

    When certified=True (default), uses:
      - separate true/fit intervals (fixes unsafe propagation bug)
      - exact positivity certification for quadratic q
      - critical-point interval updates for p=2
      - Newton fallback if certification fails
    When certified=False, uses the legacy grid-sampled approach.
    """
    lo_true = float(l0)
    hi_true = max(float(u0), lo_true * 1.0001)
    sched = []

    for t in range(T):
        lo_fit = max(lo_true, float(l_cushion))
        hi_fit = hi_true

        if kind == "affine":
            ab = fit_affine(lo_fit, hi_fit, seed=seed + t, p_val=p_val)
            a, b = ab
            if safety > 1.0:
                b = b / safety
            sched.append([a, b, lo_true, hi_true])
            lo2, hi2 = interval_update_affine([a, b], lo_true, hi_true, p_val=p_val)
        else:
            # Use local-basis fit when certified
            if certified:
                abc = fit_quadratic_local(lo_fit, hi_fit, seed=seed + t, p_val=p_val)
            else:
                abc = fit_quadratic(lo_fit, hi_fit, seed=seed + t, p_val=p_val)
            a, b, c = abc
            if safety > 1.0:
                b = b / safety
                c = c / (safety * safety)

            # Certification gate (quadratic only)
            if certified:
                pos_ok = certify_positivity_quadratic(a, b, c, lo_true, hi_true)
                if not pos_ok:
                    warnings.warn(
                        f"Step {t}: quadratic positivity certification failed, "
                        f"attempting adaptive affine fallback."
                    )

                    # Try unconstrained quadratic fallback before affine
                    abc_unconstrained = fit_quadratic(
                        lo_fit, hi_fit, seed=seed + t, p_val=p_val
                    )
                    a_unc, b_unc, c_unc = abc_unconstrained
                    if certify_positivity_quadratic(
                        a_unc, b_unc, c_unc, lo_true, hi_true
                    ):
                        a, b, c = a_unc, b_unc, c_unc
                    else:
                        # Try adaptive affine fallback before plain Newton
                        try_aff = fit_affine(lo_fit, hi_fit, seed=seed + t, p_val=p_val)
                        a_aff, b_aff = try_aff
                        # Ensure positivity for affine: exactly min(q(lo), q(hi)) > 1e-6
                        q_lo, q_hi = a_aff + b_aff * lo_true, a_aff + b_aff * hi_true
                        if min(q_lo, q_hi) > 1e-6:
                            # Success, use adaptive affine
                            a, b, c = a_aff, b_aff, 0.0
                        else:
                            warnings.warn(
                                f"Step {t}: adaptive affine fallback failed positivity, "
                                f"falling back to inverse Newton."
                            )
                            a, b, c = inverse_newton_coeffs(p_val)

            sched.append([a, b, c, lo_true, hi_true])

            # Certified or grid-based interval update
            if certified:
                lo2, hi2 = interval_update_quadratic_exact(
                    [a, b, c], lo_true, hi_true, p_val=p_val
                )
            else:
                lo2, hi2 = interval_update_quadratic(
                    [a, b, c], lo_true, hi_true, p_val=p_val
                )

        # Propagate TRUE interval (never clip to l_cushion)
        lo_true = max(1e-15, lo2)
        hi_true = max(lo_true * 1.0001, hi2)

    return sched


if __name__ == "__main__":
    l0 = 0.05
    print("Affine schedule (PE-NS), l0=0.05")
    aff = make_schedule("affine", T=3, l0=l0, l_cushion=l0, seed=0)
    for i, row in enumerate(aff, 1):
        a, b, lo, hi = row
        print(f"t={i} a={a:.10f} b={b:.10f}  interval=[{lo:.6f},{hi:.6f}]")
    print("\nQuadratic schedule (PE2/PE4), l0=0.05")
    quad = make_schedule("quad", T=4, l0=l0, l_cushion=l0, seed=0)
    for i, row in enumerate(quad, 1):
        a, b, c, lo, hi = row
        print(f"t={i} a={a:.10f} b={b:.10f} c={c:.10f} interval=[{lo:.6f},{hi:.6f}]")
