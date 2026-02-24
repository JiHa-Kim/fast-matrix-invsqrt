import math
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


def interval_update_affine(ab, lo, hi, n=16384, p_val=2):
    a, b = ab
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


def interval_update_quadratic(abc, lo, hi, n=16384, p_val=2):
    a, b, c = abc
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys + c * ys * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


def make_schedule(
    kind="affine", T=3, l0=0.05, u0=1.0, l_cushion=0.05, safety=1.0, seed=0, p_val=2
):
    lo = max(l0, l_cushion)
    hi = max(u0, lo * 1.0001)
    sched = []
    for t in range(T):
        if kind == "affine":
            ab = fit_affine(lo, hi, seed=seed + t, p_val=p_val)
            a, b = ab
            # safety: q(y/s) => a + (b/s) y
            if safety > 1.0:
                b = b / safety
            sched.append([a, b, lo, hi])
            lo2, hi2 = interval_update_affine([a, b], lo, hi, p_val=p_val)
        else:
            abc = fit_quadratic(lo, hi, seed=seed + t, p_val=p_val)
            a, b, c = abc
            if safety > 1.0:
                b = b / safety
                c = c / (safety * safety)
            sched.append([a, b, c, lo, hi])
            lo2, hi2 = interval_update_quadratic([a, b, c], lo, hi, p_val=p_val)
        lo = max(l_cushion, lo2)
        hi = max(lo * 1.0001, hi2)
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
