#!/usr/bin/env python3
import dataclasses
import torch

from .ops import (
    symmetrize,
    cuda_time_ms,
    make_spd_honest_fp64,
    init_spectrum_exact_fp64,
    apply_right_chunked,
    rel_fro,
    rel_spec,
)
from .gawlik import alpha_next, build_w_from_M, update_M, cert_action_rel_from_M
from .oracle import exact_invroot_fp64, exact_root_resid_fp64

Tensor = torch.Tensor


@dataclasses.dataclass
class RunSummary:
    success: bool
    action_rel_cert: float
    resid_M_cert: float
    resid_M_exact: float
    alpha_final: float
    alpha_pred_action_rel: float
    steps: int
    guards: int
    oracle_action_rel_fro: float
    oracle_action_rel_spec: float
    oracle_replay_rel_fro: float
    oracle_root_rel_fro: float
    oracle_root_resid: float
    ms_scale: float
    ms_small: float
    ms_apply: float
    ms_cert: float
    ms_oracle: float
    ms_total: float


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    P_storage: Tensor,
    p_root: int,
    target_action_rel: float,
    max_steps: int,
    iter_dtype: torch.dtype,
    cert_mode: str,
    exact_threshold: int,
    rhs_chunk_rows: int,
    solve_jitter_rel: float,
    oracle_mode: str,
    oracle_n_max: int,
) -> RunSummary:
    P_honest, spd_shift = make_spd_honest_fp64(P_storage, jitter_rel=solve_jitter_rel)
    n = P_honest.shape[0]

    ms_scale, (lam_min, lam_max) = cuda_time_ms(
        lambda: init_spectrum_exact_fp64(P_honest)
    )
    tau = float(lam_max)
    alpha = float((lam_min / lam_max) ** (1.0 / float(p_root)))

    M = symmetrize(P_honest / tau)
    do_oracle = oracle_mode == "on" or (oracle_mode == "auto" and n <= oracle_n_max)
    
    Z = None
    if do_oracle:
        Z = torch.eye(n, device=P_honest.device, dtype=torch.float64)
    
    B = G_storage.to(iter_dtype)

    ms_small_sum = 0.0
    ms_apply_sum = 0.0
    ms_cert_sum = 0.0
    guards = int(spd_shift > 0.0)

    final_action_rel_cert = float("inf")
    final_resid_M_cert = float("inf")
    final_resid_M_exact = float("nan")

    for it in range(1, max_steps + 1):
        ms_small, (W, _b1, mu, chol_shift) = cuda_time_ms(
            lambda: build_w_from_M(
                M=M, alpha=alpha, p=p_root, solve_jitter_rel=solve_jitter_rel
            )
        )
        ms_small_sum += ms_small
        guards += int(chol_shift > 0.0)

        ms_apply, B = cuda_time_ms(
            lambda: apply_right_chunked(B, W, rhs_chunk_rows, iter_dtype)
        )
        ms_apply_sum += ms_apply

        # Small-side exact state.
        if do_oracle:
            Z = symmetrize(W @ Z)
        M = update_M(M, W, p=p_root)
        alpha = alpha_next(alpha, mu, p=p_root)

        # Theorem-based certificate from Gawlik's Corollary 3.4 for Z_tilde A^{1/p} - I.
        eps = (1.0 - alpha) / (1.0 + alpha)
        final_action_rel_cert = float(eps / max(1.0 - eps, 1e-300))

        if final_action_rel_cert <= target_action_rel:
            break

    def cert_step_final():
        # A posteriori exact/bound diagnostic from M_tilde = Z_tilde^p P.
        # Note: M_tilde = ((1+alpha)/(2*alpha))^p * M_k
        scale_m = ((1.0 + alpha) / (2.0 * alpha)) ** p_root
        Mtilde = symmetrize(M * scale_m)

        cert_post = cert_action_rel_from_M(
            Mtilde, p=p_root, cert_mode=cert_mode, exact_threshold=exact_threshold
        )
        return cert_post

    ms_cert, cert_post = cuda_time_ms(cert_step_final)
    ms_cert_sum += ms_cert

    final_resid_M_cert = float(cert_post.resid_M_cert)
    final_resid_M_exact = float(cert_post.resid_M_exact)

    alpha_pred_action_rel = float((1.0 - alpha) / (1.0 + alpha))
    
    ms_oracle = 0.0
    oracle_action_rel_fro = float("nan")
    oracle_action_rel_spec = float("nan")
    oracle_replay_rel_fro = float("nan")
    oracle_root_rel_fro = float("nan")
    oracle_root_resid = float("nan")

    if do_oracle:
        scale = (tau**(-1.0 / float(p_root))) * ((1.0 + alpha) / (2.0 * alpha))
        Z_tilde = scale * Z
        B_tilde = B.float().to(torch.float64) * scale

        def oracle_step():
            X_exact = exact_invroot_fp64(P_honest, p=p_root)
            Y_exact = G_storage.float().to(torch.float64) @ X_exact
            Y_replay = G_storage.float().to(torch.float64) @ Z_tilde
            return (
                rel_fro(B_tilde, Y_exact),
                rel_spec(B_tilde, Y_exact),
                rel_fro(B_tilde, Y_replay),
                rel_fro(Z_tilde, X_exact),
                exact_root_resid_fp64(Z_tilde, P_honest, p=p_root),
            )

        ms_oracle, oracle_vals = cuda_time_ms(oracle_step)
        (
            oracle_action_rel_fro,
            oracle_action_rel_spec,
            oracle_replay_rel_fro,
            oracle_root_rel_fro,
            oracle_root_resid,
        ) = oracle_vals

    ms_total = ms_scale + ms_small_sum + ms_apply_sum + ms_cert_sum + ms_oracle
    return RunSummary(
        success=final_action_rel_cert <= target_action_rel,
        action_rel_cert=final_action_rel_cert,
        resid_M_cert=final_resid_M_cert,
        resid_M_exact=final_resid_M_exact,
        alpha_final=float(alpha),
        alpha_pred_action_rel=float(alpha_pred_action_rel),
        steps=it,
        guards=guards,
        oracle_action_rel_fro=float(oracle_action_rel_fro),
        oracle_action_rel_spec=float(oracle_action_rel_spec),
        oracle_replay_rel_fro=float(oracle_replay_rel_fro),
        oracle_root_rel_fro=float(oracle_root_rel_fro),
        oracle_root_resid=float(oracle_root_resid),
        ms_scale=ms_scale,
        ms_small=ms_small_sum,
        ms_apply=ms_apply_sum,
        ms_cert=ms_cert_sum,
        ms_oracle=ms_oracle,
        ms_total=ms_total,
    )
