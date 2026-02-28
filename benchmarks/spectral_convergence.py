#!/usr/bin/env python3
"""
benchmarks/spectral_convergence.py

Rigorous spectral-convergence benchmark comparing production PE-Quad schedules
against vanilla Newton-Schulz updates on SPD inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Any, List, Tuple

import torch

from fast_iroot.coeffs import _quad_coeffs, build_pe_schedules
from fast_iroot.diagnostics import (
    SpectralStepStats,
    analyze_spectral_convergence,
    format_spectral_report,
)
from benchmarks.utils import (
    get_git_metadata,
    get_repro_context,
    stable_json_sha256,
    write_text_file,
    write_json_file,
    write_sha256_sidecar,
    format_timestamp,
    repo_relative,
    write_repro_fingerprint_sidecar,
)
from benchmarks.reporting import (
    build_report_header,
    build_reproducibility_section,
    format_markdown_table,
)

# Bootstrap to allow direct script execution
try:
    from .runner import (
        ensure_repo_root_on_path,
        setup_torch_device,
        get_torch_dtype,
        get_run_directory,
    )
except ImportError:
    from runner import (
        ensure_repo_root_on_path,
        setup_torch_device,
        get_torch_dtype,
        get_run_directory,
    )

REPO_ROOT = ensure_repo_root_on_path()


def run_diagnostic_iteration(
    A_norm: torch.Tensor,
    abc_t: List[Tuple[float, float, float]],
    p_val: int,
) -> List[SpectralStepStats]:
    Y = A_norm.clone()
    stats: list[SpectralStepStats] = [analyze_spectral_convergence(Y, 0)]

    eye = torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype)
    for t, (a, b, c) in enumerate(abc_t):
        B = a * eye + b * Y
        if abs(c) > 1e-12:
            B = B + c * (Y @ Y)
        if p_val == 1:
            Y = B @ Y
        elif p_val == 2:
            Y = B @ Y @ B
        else:
            Y = torch.matrix_power(B, p_val) @ Y
        Y = 0.5 * (Y + Y.mT)
        stats.append(analyze_spectral_convergence(Y, t + 1))
    return stats


def aggregate_worst_case(
    all_stats: List[List[SpectralStepStats]],
) -> List[SpectralStepStats]:
    num_steps = len(all_stats[0])
    out: list[SpectralStepStats] = []
    for step_idx in range(num_steps):
        trials = [trial[step_idx] for trial in all_stats]
        out.append(
            SpectralStepStats(
                step=step_idx,
                min_eig=min(t.min_eig for t in trials),
                max_eig=max(t.max_eig for t in trials),
                mean_eig=sum(t.mean_eig for t in trials) / float(len(trials)),
                std_eig=0.0,
                rho_residual=max(t.rho_residual for t in trials),
                log_width=max(t.log_width for t in trials),
                error_to_identity=max(t.error_to_identity for t in trials),
                clustering_90=min(t.clustering_90 for t in trials),
                clustering_99=min(t.clustering_99 for t in trials),
            )
        )
    return out


def _stats_to_json_rows(stats: List[SpectralStepStats]) -> list[dict[str, Any]]:
    return [asdict(s) for s in stats]


def _build_markdown(
    *,
    args: argparse.Namespace,
    device: torch.device,
    pe_steps: int,
    coeffs: list[tuple[float, float, float]],
    worst_pe: list[SpectralStepStats],
    worst_ns: list[SpectralStepStats],
    out_path: str,
    json_path: str,
    manifest_path: str,
) -> str:
    config = vars(args).copy()
    config["device"] = str(device)
    config["pe_steps"] = pe_steps

    lines = build_report_header("Spectral Convergence Benchmark", config)

    lines.extend(
        [
            "## Column Definitions",
            "",
            "- **Min Î» / Max Î»**: Minimum and maximum eigenvalues of the current iterate.",
            "- **Ï(I-Y)**: Spectral radius of the residual matrix, $\\rho(I - Y) = \\max_i |1 - \\lambda_i|$. Measures overall closeness to identity.",
            "- **log(M/m)**: Log-width of the spectral interval, $\\log(\\lambda_{\\max}/\\lambda_{\\min})$. Primary indicator of iteration progress for coupled methods.",
            "- **C90% / C99%**: Fraction of eigenvalues clustered within 10% and 1% of identity (1.0).",
            "",
        ]
    )

    lines.append("## Coefficients (PE-Quad)")
    lines.append("")
    coeff_headers = ["Step", "a", "b", "c"]
    coeff_rows = [[i, a, b, c] for i, (a, b, c) in enumerate(coeffs)]
    lines.append(format_markdown_table(coeff_headers, coeff_rows))
    lines.append("")

    lines.append("## PE-Quad (Worst Case Over Trials)")
    lines.append("")
    lines.append(format_spectral_report(worst_pe))
    lines.append("")

    lines.append("## Newton-Schulz (Worst Case Over Trials)")
    lines.append("")
    lines.append(format_spectral_report(worst_ns))
    lines.append("")

    lines.extend(
        build_reproducibility_section(json_path, manifest_path, str(REPO_ROOT))
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run spectral convergence benchmark")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--coeff-mode", type=str, default="precomputed")
    p.add_argument("--coeff-seed", type=int, default=0)
    p.add_argument("--coeff-safety", type=float, default=1.0)
    p.add_argument("--coeff-no-final-safety", action="store_true")
    p.add_argument("--run-name", type=str, default="spectral_convergence")
    p.add_argument("--out", type=str, default="")
    p.add_argument(
        "--prod",
        action="store_true",
        help="Update production documentation by default.",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--manifest-out", type=str, default="")
    p.add_argument(
        "--integrity-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write .sha256 sidecars for all generated artifacts (default: true).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n) < 2:
        raise ValueError("--n must be >= 2")
    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")

    device = setup_torch_device(str(args.device))
    dtype = get_torch_dtype(str(args.dtype))
    g = torch.Generator(device=device).manual_seed(int(args.seed))

    rel_run_dir, abs_run_dir = get_run_directory(str(args.run_name), REPO_ROOT)

    # Determine output paths
    if args.prod:
        out_path = os.path.join(
            REPO_ROOT, "docs", "benchmarks", "spectral_convergence_production.md"
        )
    else:
        out_path = str(args.out).strip() or os.path.join(
            abs_run_dir, "spectral_convergence.md"
        )

    json_path = str(args.json_out).strip() or os.path.join(
        abs_run_dir, "spectral_convergence.json"
    )
    manifest_path = str(args.manifest_out).strip() or os.path.join(
        abs_run_dir, "spectral_manifest.json"
    )

    pe_sched, _ = build_pe_schedules(
        l_target=float(args.l_target),
        device=device,
        coeff_mode=str(args.coeff_mode),
        coeff_seed=int(args.coeff_seed),
        coeff_safety=float(args.coeff_safety),
        coeff_no_final_safety=bool(args.coeff_no_final_safety),
        p_val=int(args.p),
    )
    abc_pe = _quad_coeffs(pe_sched)
    abc_ns = [((float(args.p) + 1.0) / float(args.p), -1.0 / float(args.p), 0.0)] * len(
        abc_pe
    )

    all_pe_stats: list[list[SpectralStepStats]] = []
    all_ns_stats: list[list[SpectralStepStats]] = []

    print(
        f"Running spectral convergence: n={args.n}, p={args.p}, trials={args.trials}, "
        f"device={device}, dtype={dtype}"
    )
    for trial in range(int(args.trials)):
        e = torch.linspace(
            float(args.l_target),
            1.0,
            steps=int(args.n),
            device=device,
            dtype=dtype,
        )
        Q, _ = torch.linalg.qr(
            torch.randn(
                int(args.n), int(args.n), device=device, dtype=dtype, generator=g
            )
        )
        A_norm = (Q * e.unsqueeze(0)) @ Q.mT
        all_pe_stats.append(run_diagnostic_iteration(A_norm, abc_pe, int(args.p)))
        all_ns_stats.append(run_diagnostic_iteration(A_norm, abc_ns, int(args.p)))
        print(f"  trial {trial + 1}/{args.trials} complete")

    worst_pe = aggregate_worst_case(all_pe_stats)
    worst_ns = aggregate_worst_case(all_ns_stats)

    report = _build_markdown(
        args=args,
        device=device,
        pe_steps=len(abc_pe),
        coeffs=abc_pe,
        worst_pe=worst_pe,
        worst_ns=worst_ns,
        out_path=out_path,
        json_path=json_path,
        manifest_path=manifest_path,
    )
    write_text_file(out_path, report)

    raw_payload: dict[str, Any] = {
        "schema": "spectral_convergence.v1",
        "generated_at": format_timestamp(),
        "run_dir": rel_run_dir.replace("\\", "/"),
        "argv": list(sys.argv[1:]),
        "args": vars(args),
        "hash_algorithm": "sha256",
        "git": get_git_metadata(str(REPO_ROOT)),
        "pe_steps": len(abc_pe),
        "pe_coeffs": [{"a": a, "b": b, "c": c} for (a, b, c) in abc_pe],
        "worst_case": {
            "pe_quad": _stats_to_json_rows(worst_pe),
            "newton_schulz": _stats_to_json_rows(worst_ns),
        },
    }
    write_json_file(json_path, raw_payload)

    repro_context = get_repro_context(str(REPO_ROOT), args)
    manifest: dict[str, Any] = {
        "schema": "spectral_manifest.v1",
        "generated_at": format_timestamp(),
        "run_dir": rel_run_dir.replace("\\", "/"),
        "outputs": {
            "markdown": repo_relative(out_path, str(REPO_ROOT)),
            "json": repo_relative(json_path, str(REPO_ROOT)),
        },
        "repro_context": repro_context,
        "repro_fingerprint_sha256": stable_json_sha256(repro_context),
    }
    write_json_file(manifest_path, manifest)

    checksum_paths: list[str] = []
    if bool(args.integrity_checksums):
        for pth in (out_path, json_path, manifest_path):
            write_sha256_sidecar(pth)
            checksum_paths.append(f"{pth}.sha256")
        repro_sidecar = write_repro_fingerprint_sidecar(
            manifest_path, manifest["repro_fingerprint_sha256"]
        )
        checksum_paths.append(repro_sidecar)

    print(f"Wrote report: {repo_relative(out_path, str(REPO_ROOT))}")
    print(f"Wrote raw JSON: {repo_relative(json_path, str(REPO_ROOT))}")
    print(f"Wrote manifest: {repo_relative(manifest_path, str(REPO_ROOT))}")
    print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
    if checksum_paths:
        print("Wrote checksum sidecars:")
        for pth in checksum_paths:
            print(f"  - {repo_relative(pth, str(REPO_ROOT))}")


if __name__ == "__main__":
    main()
