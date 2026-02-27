from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from fast_iroot import PrecondStats, precond_spd
from fast_iroot.metrics import compute_quality_stats, iroot_relative_error

from benchmarks.common import median, pctl, time_ms_repeat, time_ms_any

MATRIX_IROOT_METHODS: List[str] = ["Inverse-Newton", "PE-Quad", "PE-Quad-Coupled"]


@dataclass
class BenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    residual: float
    residual_p95: float
    residual_max: float
    residual_spec: float
    sym_x: float
    sym_w: float
    mv_err: float
    hard_dir: float
    relerr: float
    bad: int
    mem_alloc_mb: float
    mem_reserved_mb: float
    coupled_y_resid: float


@dataclass
class PreparedInput:
    A_norm: torch.Tensor
    stats: PrecondStats


@torch.no_grad()
def prepare_preconditioned_inputs(
    mats: List[torch.Tensor],
    device: torch.device,
    precond: str,
    precond_ruiz_iters: int,
    ridge_rel: float,
    l_target: float,
) -> Tuple[List[PreparedInput], float]:
    prepared: List[PreparedInput] = []
    ms_pre_list: List[float] = []
    for A in mats:
        t_pre, out = time_ms_any(
            lambda: precond_spd(
                A,
                mode=precond,
                ruiz_iters=precond_ruiz_iters,
                ridge_rel=ridge_rel,
                l_target=l_target,
                compute_rho_proxy=False,
            ),
            device,
        )
        A_norm, stats = out
        ms_pre_list.append(t_pre)
        prepared.append(PreparedInput(A_norm=A_norm, stats=stats))
    return prepared, (median(ms_pre_list) if ms_pre_list else float("nan"))


def _build_runner(
    method: str,
    pe_quad_coeffs: Sequence[Tuple[float, float, float]],
    symmetrize_Y: bool,
    symmetrize_every: int,
    p_val: int,
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> Callable[[torch.Tensor, Optional[object]], Tuple[torch.Tensor, object]]:
    if method == "Inverse-Newton":
        a = (p_val + 1.0) / p_val
        b = -1.0 / p_val
        c = 0.0
        inv_newton_coeffs = [(a, b, c)] * len(pe_quad_coeffs)

        def run(A_norm: torch.Tensor, ws: Optional[object]):
            return coupled_fn(
                A_norm,
                abc_t=inv_newton_coeffs,
                p_val=p_val,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                symmetrize_every=symmetrize_every,
                terminal_last_step=True,
            )

        return run

    if method == "PE-Quad":

        def run(A_norm: torch.Tensor, ws: Optional[object]):
            return uncoupled_fn(
                A_norm,
                abc_t=pe_quad_coeffs,
                p_val=p_val,
                ws=ws,
                symmetrize_X=symmetrize_Y,
            )

        return run

    if method == "PE-Quad-Coupled":

        def run(A_norm: torch.Tensor, ws: Optional[object]):
            return coupled_fn(
                A_norm,
                abc_t=pe_quad_coeffs,
                p_val=p_val,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                symmetrize_every=symmetrize_every,
                terminal_last_step=True,
            )

        return run

    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def eval_method(
    prepared_inputs: List[PreparedInput],
    ms_precond_median: float,
    device: torch.device,
    method: str,
    pe_quad_coeffs: Sequence[Tuple[float, float, float]],
    timing_reps: int,
    symmetrize_Y: bool,
    symmetrize_every: int,
    compute_relerr: bool,
    power_iters: int,
    mv_samples: int,
    hard_probe_iters: int,
    p_val: int,
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> BenchResult:
    ms_iter_list: List[float] = []
    res_list: List[float] = []
    res2_list: List[float] = []
    symx_list: List[float] = []
    symw_list: List[float] = []
    mv_list: List[float] = []
    hard_list: List[float] = []
    err_list: List[float] = []
    mem_alloc_list: List[float] = []
    mem_res_list: List[float] = []
    y_res_list: List[float] = []
    bad = 0

    if len(prepared_inputs) == 0:
        return BenchResult(
            ms=float("nan"),
            ms_iter=float("nan"),
            ms_precond=float("nan"),
            residual=float("nan"),
            residual_p95=float("nan"),
            residual_max=float("nan"),
            residual_spec=float("nan"),
            sym_x=float("nan"),
            sym_w=float("nan"),
            mv_err=float("nan"),
            hard_dir=float("nan"),
            relerr=float("nan"),
            bad=0,
            mem_alloc_mb=float("nan"),
            mem_reserved_mb=float("nan"),
            coupled_y_resid=float("nan"),
        )

    runner = _build_runner(
        method=method,
        pe_quad_coeffs=pe_quad_coeffs,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        p_val=p_val,
        uncoupled_fn=uncoupled_fn,
        coupled_fn=coupled_fn,
    )

    ws: Optional[object] = None
    eye_mat: Optional[torch.Tensor] = None

    for prep in prepared_inputs:
        A_norm = prep.A_norm
        n = A_norm.shape[-1]

        if (
            eye_mat is None
            or eye_mat.shape != (n, n)
            or eye_mat.device != A_norm.device
        ):
            eye_mat = torch.eye(n, device=A_norm.device, dtype=torch.float32)

        ws = None

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
            _, ws_mem = runner(A_norm, ws)
            mem_alloc_list.append(
                torch.cuda.max_memory_allocated(device=device) / (1024**2)
            )
            mem_res_list.append(
                torch.cuda.max_memory_reserved(device=device) / (1024**2)
            )
            del ws_mem
            ws = None

        def run_once() -> torch.Tensor:
            nonlocal ws
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            Xn, ws = runner(A_norm, ws)
            return Xn

        ms_iter, Xn = time_ms_repeat(run_once, device, reps=timing_reps)
        ms_iter_list.append(ms_iter)

        if not torch.isfinite(Xn).all():
            bad += 1
            for lst in [
                res_list,
                res2_list,
                symx_list,
                symw_list,
                mv_list,
                hard_list,
                err_list,
                y_res_list,
            ]:
                lst.append(float("inf"))
            continue

        if ws is not None and hasattr(ws, "Y"):
            y_res = float(torch.linalg.matrix_norm(ws.Y - eye_mat, ord="fro").mean())
            y_res_list.append(y_res)
        else:
            y_res_list.append(float("nan"))

        q = compute_quality_stats(
            Xn,
            A_norm,
            power_iters=power_iters,
            mv_samples=mv_samples,
            hard_probe_iters=hard_probe_iters,
            eye_mat=eye_mat,
            p_val=p_val,
        )
        res_list.append(q.residual_fro)
        res2_list.append(q.residual_spec)
        symx_list.append(q.sym_x)
        symw_list.append(q.sym_w)
        mv_list.append(q.mv_err)
        hard_list.append(q.hard_dir_err)

        if compute_relerr:
            err_list.append(
                float(
                    iroot_relative_error(Xn.float(), A_norm.float(), p_val=p_val)
                    .mean()
                    .item()
                )
            )
        else:
            err_list.append(float("nan"))

    ms_iter_med = median(ms_iter_list)
    ms_pre_med = ms_precond_median

    return BenchResult(
        ms=ms_pre_med + ms_iter_med,
        ms_iter=ms_iter_med,
        ms_precond=ms_pre_med,
        residual=median(res_list),
        residual_p95=pctl(res_list, 0.95),
        residual_max=max(float(x) for x in res_list) if res_list else float("nan"),
        residual_spec=median(res2_list),
        sym_x=median(symx_list),
        sym_w=median(symw_list),
        mv_err=median(mv_list),
        hard_dir=median(hard_list),
        relerr=median(err_list),
        bad=bad,
        mem_alloc_mb=median(mem_alloc_list) if mem_alloc_list else float("nan"),
        mem_reserved_mb=median(mem_res_list) if mem_res_list else float("nan"),
        coupled_y_resid=median(y_res_list) if y_res_list else float("nan"),
    )
