#!/usr/bin/env python3
"""
benchmarks/spectral_convergence.py

Rigorous spectral-convergence benchmark comparing production PE-Quad schedules
against vanilla Newton-Schulz updates on SPD inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any, List, Tuple

import torch

try:
    from benchmarks._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fast_iroot.coeffs import _quad_coeffs, build_pe_schedules
from fast_iroot.diagnostics import (
    SpectralStepStats,
    analyze_spectral_convergence,
    format_spectral_report,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_sha_sidecar(path: str, digest: str) -> str:
    sidecar = f"{path}.sha256"
    _write_text(sidecar, f"{digest}  {os.path.basename(path)}\n")
    return sidecar


def _stable_json_sha(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _git_meta() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str | None:
        try:
            out = subprocess.check_output(
                cmd,
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or None
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": (_run(["git", "status", "--porcelain"]) or "") != "",
    }


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


def aggregate_worst_case(all_stats: List[List[SpectralStepStats]]) -> List[SpectralStepStats]:
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
) -> str:
    lines: list[str] = []
    lines.append("# Spectral Convergence Benchmark")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- n: `{int(args.n)}`")
    lines.append(f"- p: `{int(args.p)}`")
    lines.append(f"- trials: `{int(args.trials)}`")
    lines.append(f"- l_target: `{float(args.l_target)}`")
    lines.append(f"- dtype: `{str(args.dtype)}`")
    lines.append(f"- device: `{str(device)}`")
    lines.append(f"- seed: `{int(args.seed)}`")
    lines.append(f"- coeff_mode: `{str(args.coeff_mode)}`")
    lines.append(f"- coeff_seed: `{int(args.coeff_seed)}`")
    lines.append(f"- coeff_safety: `{float(args.coeff_safety)}`")
    lines.append(f"- coeff_no_final_safety: `{bool(args.coeff_no_final_safety)}`")
    lines.append(f"- pe_steps: `{pe_steps}`")
    lines.append("")
    lines.append("## Coefficients (PE-Quad)")
    lines.append("")
    lines.append("| Step | a | b | c |")
    lines.append("|---:|---:|---:|---:|")
    for idx, (a, b, c) in enumerate(coeffs):
        lines.append(f"| {idx} | {a:.9f} | {b:.9f} | {c:.9f} |")
    lines.append("")
    lines.append("## PE-Quad (Worst Case Over Trials)")
    lines.append("")
    lines.append(format_spectral_report(worst_pe))
    lines.append("")
    lines.append("## Newton-Schulz (Worst Case Over Trials)")
    lines.append("")
    lines.append(format_spectral_report(worst_ns))
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("This report is paired with:")
    lines.append("- `spectral_convergence.json` (raw per-step rows)")
    lines.append("- `spectral_manifest.json` (run metadata + reproducibility fingerprint)")
    lines.append("- `.sha256` sidecars for all output files")
    lines.append("")
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

    if str(args.device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable")

    dtype = torch.float64 if str(args.dtype) == "fp64" else torch.float32
    g = torch.Generator(device=device).manual_seed(int(args.seed))

    today = datetime.now().strftime("%Y_%m_%d")
    now = datetime.now().strftime("%H%M%S")
    run_name = str(args.run_name).strip() or "spectral_convergence"
    run_dir_rel = os.path.join("benchmark_results", "runs", today, f"{now}_{run_name}")
    run_dir_abs = os.path.join(REPO_ROOT, run_dir_rel)

    out_path = str(args.out).strip() or os.path.join(run_dir_abs, "spectral_convergence.md")
    json_path = str(args.json_out).strip() or os.path.join(run_dir_abs, "spectral_convergence.json")
    manifest_path = str(args.manifest_out).strip() or os.path.join(run_dir_abs, "spectral_manifest.json")

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
    abc_ns = [((float(args.p) + 1.0) / float(args.p), -1.0 / float(args.p), 0.0)] * len(abc_pe)

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
            torch.randn(int(args.n), int(args.n), device=device, dtype=dtype, generator=g)
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
    )
    _write_text(out_path, report)

    raw_payload: dict[str, Any] = {
        "schema": "spectral_convergence.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir_rel.replace("\\", "/"),
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "pe_steps": len(abc_pe),
        "pe_coeffs": [{"a": a, "b": b, "c": c} for (a, b, c) in abc_pe],
        "worst_case": {
            "pe_quad": _stats_to_json_rows(worst_pe),
            "newton_schulz": _stats_to_json_rows(worst_ns),
        },
    }
    _write_json(json_path, raw_payload)

    repro_context = {
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "git": _git_meta(),
    }
    manifest: dict[str, Any] = {
        "schema": "spectral_manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir_rel.replace("\\", "/"),
        "outputs": {
            "markdown": os.path.relpath(out_path, REPO_ROOT).replace("\\", "/"),
            "json": os.path.relpath(json_path, REPO_ROOT).replace("\\", "/"),
        },
        "repro_context": repro_context,
        "repro_fingerprint_sha256": _stable_json_sha(repro_context),
    }
    _write_json(manifest_path, manifest)

    checksum_paths: list[str] = []
    if bool(args.integrity_checksums):
        for pth in (out_path, json_path, manifest_path):
            checksum_paths.append(_write_sha_sidecar(pth, _sha256_file(pth)))
        repro_sidecar = f"{manifest_path}.repro.sha256"
        _write_text(
            repro_sidecar,
            f"{manifest['repro_fingerprint_sha256']}  reproducibility_fingerprint\n",
        )
        checksum_paths.append(repro_sidecar)

    print(f"Wrote report: {os.path.relpath(out_path, REPO_ROOT)}")
    print(f"Wrote raw JSON: {os.path.relpath(json_path, REPO_ROOT)}")
    print(f"Wrote manifest: {os.path.relpath(manifest_path, REPO_ROOT)}")
    print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
    if checksum_paths:
        print("Wrote checksum sidecars:")
        for pth in checksum_paths:
            print(f"  - {os.path.relpath(pth, REPO_ROOT).replace('\\', '/')}")


if __name__ == "__main__":
    main()
