#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix.

Modes:
- default: writes per-run .txt logs under
  benchmark_results/runs/<timestamp>_solver_benchmarks/*_solve_logs/
- --markdown: writes one organized markdown report file (--out)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str  # "spd" | "nonspd"
    cmd: list[str]
    txt_out: str


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _run_and_capture(cmd: list[str]) -> str:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result.stdout


def _run_and_write_txt(cmd: list[str], out_path: str) -> None:
    out = _run_and_capture(cmd)
    print(f"Logging to: {out_path}")
    _write_text_file(out_path, out)


def _write_text_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json_file(path: str, payload: dict[str, Any]) -> None:
    _write_text_file(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _stable_json_sha256(payload: Any) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canon)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_sidecar(path: str, digest: str) -> str:
    sidecar = f"{path}.sha256"
    line = f"{digest}  {os.path.basename(path)}\n"
    _write_text_file(sidecar, line)
    return sidecar


def _write_repro_fingerprint_sidecar(manifest_path: str, fingerprint: str) -> str:
    sidecar = f"{manifest_path}.repro.sha256"
    line = f"{fingerprint}  reproducibility_fingerprint\n"
    _write_text_file(sidecar, line)
    return sidecar


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def _capture_optional(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() if res.stdout is not None else None


def _tracked_source_sha256() -> str | None:
    """Hash current tracked source content (working tree, not just HEAD commit)."""
    try:
        res = subprocess.run(
            ["git", "ls-files", "-z"],
            capture_output=True,
            cwd=REPO_ROOT,
            check=False,
        )
    except Exception:
        return None
    if res.returncode != 0 or not isinstance(res.stdout, (bytes, bytearray)):
        return None

    paths = [
        p.decode("utf-8", errors="replace")
        for p in bytes(res.stdout).split(b"\x00")
        if p
    ]
    h = hashlib.sha256()
    for rel_path in sorted(paths):
        norm_rel = rel_path.replace("\\", "/")
        abs_path = os.path.join(REPO_ROOT, rel_path)
        h.update(norm_rel.encode("utf-8"))
        h.update(b"\x00")
        try:
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        except Exception:
            # Keep hash deterministic even if a tracked file disappears mid-run.
            h.update(b"<missing>")
        h.update(b"\x00")
    return h.hexdigest()


def _git_meta() -> dict[str, Any]:
    head = _capture_optional(["git", "rev-parse", "HEAD"])
    branch = _capture_optional(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _capture_optional(["git", "status", "--short"])
    tracked_sha = _tracked_source_sha256()
    return {
        "head": head,
        "branch": branch,
        "dirty": bool(status),
        "tracked_source_sha256": tracked_sha,
    }


def _base_manifest(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "repo_root": REPO_ROOT.replace("\\", "/"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv[1:]),
        "args": vars(args),
        "hash_algorithm": "sha256",
        "git": _git_meta(),
    }


def _repro_context(
    *,
    mode: str,
    specs: list[RunSpec],
    extra_args: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
    return {
        "mode": mode,
        "argv": list(sys.argv[1:]),
        "args": vars(args),
        "specs": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "cmd_base": spec.cmd,
            }
            for spec in specs
        ],
        "extra_args": extra_args,
        "git": _git_meta(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "env": env,
    }


def _parse_csv_tokens(spec: str) -> list[str]:
    return [tok.strip() for tok in str(spec).split(",") if tok.strip()]


def _split_extra_args(spec: str) -> list[str]:
    text = str(spec).strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def _filter_specs(specs: Iterable[RunSpec], only_tokens: list[str]) -> list[RunSpec]:
    toks = [t.lower() for t in only_tokens if t]
    if not toks:
        return list(specs)
    out: list[RunSpec] = []
    for spec in specs:
        hay = f"{spec.name} {spec.txt_out}".lower()
        if any(tok in hay for tok in toks):
            out.append(spec)
    return out


def _build_specs(
    trials: int,
    dtype: str,
    timing_reps: int,
    warmup_reps: int,
    *,
    spd_dir: str,
    nonspd_dir: str,
    ts: str,
) -> list[RunSpec]:
    specs: list[RunSpec] = []

    # SPD, p in {1,2,4}, k<n (n={1024}, k={1,16,64})
    for p_val in (1, 2, 4):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve",
            "--p",
            str(p_val),
            "--sizes",
            "1024",
            "--k",
            "1,16,64",
            "--trials",
            str(trials),
            "--timing-reps",
            str(timing_reps),
            "--timing-warmup-reps",
            str(warmup_reps),
            "--dtype",
            dtype,
        ]
        specs.append(
            RunSpec(
                name=f"SPD p={p_val} k<n",
                kind="spd",
                cmd=cmd,
                txt_out=os.path.join(
                    spd_dir,
                    f"{ts}_spd_p{p_val}_klt_n_sizes1024_k1_16_64.txt",
                ),
            )
        )

    # SPD, p in {1,2,4}, k=n for n in {256,512,1024}
    for p_val in (1, 2, 4):
        for n_val in (256, 512, 1024):
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "benchmarks.solve.matrix_solve",
                "--p",
                str(p_val),
                "--sizes",
                str(n_val),
                "--k",
                str(n_val),
                "--trials",
                str(trials),
                "--timing-reps",
                str(timing_reps),
                "--timing-warmup-reps",
                str(warmup_reps),
                "--dtype",
                dtype,
            ]
            specs.append(
                RunSpec(
                    name=f"SPD p={p_val} k=n={n_val}",
                    kind="spd",
                    cmd=cmd,
                    txt_out=os.path.join(
                        spd_dir,
                        f"{ts}_spd_p{p_val}_keq_n_n{n_val}_k{n_val}.txt",
                    ),
                )
            )

    # non-SPD p=1, k<n (n={1024}, k={1,16,64})
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "benchmarks.solve.matrix_solve_nonspd",
        "--p",
        "1",
        "--sizes",
        "1024",
        "--k",
        "1,16,64",
        "--trials",
        str(trials),
        "--timing-reps",
        str(timing_reps),
        "--timing-warmup-reps",
        str(warmup_reps),
        "--dtype",
        dtype,
    ]
    specs.append(
        RunSpec(
            name="non-SPD p=1 k<n",
            kind="nonspd",
            cmd=cmd,
            txt_out=os.path.join(
                nonspd_dir,
                f"{ts}_nonspd_p1_klt_n_sizes1024_k1_16_64.txt",
            ),
        )
    )

    # non-SPD p=1, k=n for n in {256,512,1024}
    for n_val in (256, 512, 1024):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve_nonspd",
            "--p",
            "1",
            "--sizes",
            str(n_val),
            "--k",
            str(n_val),
            "--trials",
            str(trials),
            "--timing-reps",
            str(timing_reps),
            "--timing-warmup-reps",
            str(warmup_reps),
            "--dtype",
            dtype,
        ]
        specs.append(
            RunSpec(
                name=f"non-SPD p=1 k=n={n_val}",
                kind="nonspd",
                cmd=cmd,
                txt_out=os.path.join(
                    nonspd_dir,
                    f"{ts}_nonspd_p1_keq_n_n{n_val}_k{n_val}.txt",
                ),
            )
        )

    return specs


def _parse_rows(
    raw: str, kind: str
) -> list[tuple[str, int, int, str, str, float, float, float]]:
    rows: list[tuple[str, int, int, str, str, float, float, float]] = []
    current_n = -1
    current_k = -1
    current_case = ""

    hdr_re = re.compile(r"==\s+(?:SPD|Non-SPD)\s+Size\s+(\d+)x\1\s+\|\s+RHS\s+\1x(\d+)")
    case_re = re.compile(r"^--\s+case\s+([^\s]+)\s+--")
    line_re = re.compile(
        r"^(.*?)\s+(\d+\.\d+)\s+ms\s+\(pre\s+(\d+\.\d+)\s+\+\s+iter\s+(\d+\.\d+)\).*?"
        r"relerr\s+vs\s+(?:true|solve):\s+([0-9.eE+-]+)"
    )

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        hm = hdr_re.search(line)
        if hm:
            current_n = int(hm.group(1))
            current_k = int(hm.group(2))
            continue
        cm = case_re.match(line)
        if cm:
            current_case = cm.group(1)
            continue
        lm = line_re.match(line)
        if lm and current_n > 0 and current_k > 0 and current_case:
            method = lm.group(1).strip()
            total_ms = float(lm.group(2))
            iter_ms = float(lm.group(4))
            relerr = float(lm.group(5))
            rows.append(
                (
                    kind,
                    current_n,
                    current_k,
                    current_case,
                    method,
                    total_ms,
                    iter_ms,
                    relerr,
                )
            )

    return rows


def _to_markdown(all_rows: list[tuple[str, int, int, str, str, float, float, float]]) -> str:
    out: list[str] = []
    out.append("# Solver Benchmark Report")
    out.append("")
    out.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append("")
    out.append("| kind | n | k | case | method | total_ms | iter_ms | relerr |")
    out.append("|---|---:|---:|---|---|---:|---:|---:|")

    for row in sorted(all_rows, key=lambda r: (r[0], r[1], r[2], r[3], r[4])):
        kind, n, k, case_name, method, total_ms, iter_ms, relerr = row
        out.append(
            f"| {kind} | {n} | {k} | {case_name} | {method} | {total_ms:.3f} | {iter_ms:.3f} | {relerr:.3e} |"
        )
    out.append("")
    return "\n".join(out)


def _to_markdown_ab(
    rows_a: list[tuple[str, int, int, str, str, float, float, float]],
    rows_b: list[tuple[str, int, int, str, str, float, float, float]],
    *,
    label_a: str,
    label_b: str,
) -> str:
    def _key(row: tuple[str, int, int, str, str, float, float, float]):
        return row[0], row[1], row[2], row[3], row[4]

    map_a = {_key(r): r for r in rows_a}
    map_b = {_key(r): r for r in rows_b}
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(keys) == 0:
        raise RuntimeError(
            "A/B rows had no overlapping keys; cannot build comparable report."
        )

    out: list[str] = []
    out.append("# Solver Benchmark A/B Report")
    out.append("")
    out.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append("")
    out.append(f"A: {label_a}")
    out.append(f"B: {label_b}")
    out.append("")
    out.append(
        "| kind | n | k | case | method | "
        f"{label_a}_total_ms | {label_b}_total_ms | delta_ms(B-A) | delta_pct | "
        f"{label_a}_iter_ms | {label_b}_iter_ms | "
        f"{label_a}_relerr | {label_b}_relerr | relerr_ratio(B/A) |"
    )
    out.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for key in keys:
        ra = map_a[key]
        rb = map_b[key]
        kind, n, k, case_name, method = key
        a_total, a_iter, a_rel = ra[5], ra[6], ra[7]
        b_total, b_iter, b_rel = rb[5], rb[6], rb[7]
        d_ms = b_total - a_total
        d_pct = (100.0 * d_ms / a_total) if a_total != 0 else float("nan")
        rel_ratio = (b_rel / a_rel) if a_rel != 0 else float("nan")
        out.append(
            f"| {kind} | {n} | {k} | {case_name} | {method} | "
            f"{a_total:.3f} | {b_total:.3f} | {d_ms:.3f} | {d_pct:.2f}% | "
            f"{a_iter:.3f} | {b_iter:.3f} | {a_rel:.3e} | {b_rel:.3e} | {rel_ratio:.3f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maintained solver benchmark suites")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--timing-reps", type=int, default=5)
    parser.add_argument("--timing-warmup-reps", type=int, default=2)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=(
            "Optional comma-separated substring filters for selecting a subset of run specs "
            "(matches spec name and output filename)."
        ),
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help=(
            "Extra CLI args appended to every benchmark command; useful for focused A/B runs "
            "without editing this script."
        ),
    )
    parser.add_argument("--markdown", action="store_true", help="Write one markdown report instead of per-run txt files")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output markdown path used with --markdown",
    )
    parser.add_argument(
        "--ab-extra-args-a",
        type=str,
        default="",
        help="A-side extra args (appended to every benchmark command) for A/B markdown compare mode.",
    )
    parser.add_argument(
        "--ab-extra-args-b",
        type=str,
        default="",
        help="B-side extra args (appended to every benchmark command) for A/B markdown compare mode.",
    )
    parser.add_argument(
        "--ab-label-a",
        type=str,
        default="A",
        help="Display label for A-side runs in A/B markdown report.",
    )
    parser.add_argument(
        "--ab-label-b",
        type=str,
        default="B",
        help="Display label for B-side runs in A/B markdown report.",
    )
    parser.add_argument(
        "--ab-out",
        type=str,
        default="",
        help="Output markdown path used for A/B compare mode.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default="",
        help="Output JSON manifest path (run metadata + reproducibility fingerprint).",
    )
    parser.add_argument(
        "--integrity-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also write SHA256 checksums for output artifacts (.txt/.md/.json) "
            "to detect tampering/corruption. This is integrity, not reproducibility."
        ),
    )
    args = parser.parse_args()

    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")
    if int(args.timing_reps) < 1:
        raise ValueError("--timing-reps must be >= 1")
    if int(args.timing_warmup_reps) < 0:
        raise ValueError("--timing-warmup-reps must be >= 0")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_rel = os.path.join(
        "benchmark_results", "runs", f"{run_ts}_solver_benchmarks"
    )
    run_dir_abs = os.path.join(REPO_ROOT, run_dir_rel)
    spd_dir_abs = os.path.join(run_dir_abs, "spd_solve_logs")
    nonspd_dir_abs = os.path.join(run_dir_abs, "nonspd_solve_logs")
    _ensure_dirs(spd_dir_abs, nonspd_dir_abs)

    if not str(args.out).strip():
        args.out = os.path.join(run_dir_rel, "solver_benchmarks.md")
    if not str(args.ab_out).strip():
        args.ab_out = os.path.join(run_dir_rel, "solver_benchmarks_ab.md")
    if not str(args.manifest_out).strip():
        args.manifest_out = os.path.join(run_dir_rel, "run_manifest.json")

    specs_all = _build_specs(
        trials=int(args.trials),
        dtype=str(args.dtype),
        timing_reps=int(args.timing_reps),
        warmup_reps=int(args.timing_warmup_reps),
        spd_dir=spd_dir_abs,
        nonspd_dir=nonspd_dir_abs,
        ts=run_ts,
    )
    specs = _filter_specs(specs_all, _parse_csv_tokens(args.only))
    if len(specs) == 0:
        raise ValueError("No benchmark specs matched --only filter")

    base_extra_args = _split_extra_args(args.extra_args)
    ab_mode = bool(str(args.ab_extra_args_a).strip() or str(args.ab_extra_args_b).strip())
    integrity_checksums = bool(args.integrity_checksums)

    manifest_path = os.path.join(REPO_ROOT, str(args.manifest_out))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    if ab_mode:
        a_extra = base_extra_args + _split_extra_args(args.ab_extra_args_a)
        b_extra = base_extra_args + _split_extra_args(args.ab_extra_args_b)

        rows_a: list[tuple[str, int, int, str, str, float, float, float]] = []
        rows_b: list[tuple[str, int, int, str, str, float, float, float]] = []
        run_records: list[dict[str, Any]] = []

        for spec in specs:
            cmd_a = spec.cmd + a_extra
            raw_a = _run_and_capture(cmd_a)
            rows_a_spec = _parse_rows(raw_a, spec.kind)
            rows_a.extend(rows_a_spec)
            run_records.append(
                {
                    "spec_name": spec.name,
                    "kind": spec.kind,
                    "variant": "A",
                    "cmd": cmd_a,
                    "stdout_sha256": _sha256_text(raw_a),
                    "parsed_rows": len(rows_a_spec),
                }
            )

            cmd_b = spec.cmd + b_extra
            raw_b = _run_and_capture(cmd_b)
            rows_b_spec = _parse_rows(raw_b, spec.kind)
            rows_b.extend(rows_b_spec)
            run_records.append(
                {
                    "spec_name": spec.name,
                    "kind": spec.kind,
                    "variant": "B",
                    "cmd": cmd_b,
                    "stdout_sha256": _sha256_text(raw_b),
                    "parsed_rows": len(rows_b_spec),
                }
            )

        if len(rows_a) == 0 or len(rows_b) == 0:
            raise RuntimeError(
                "A/B mode produced no parsed rows; check benchmark output format/regex."
            )

        out_path = os.path.join(REPO_ROOT, args.ab_out)
        md_text = _to_markdown_ab(
            rows_a,
            rows_b,
            label_a=str(args.ab_label_a),
            label_b=str(args.ab_label_b),
        )
        _write_text_file(out_path, md_text)
        out_sidecar: str | None = None
        out_sha: str | None = None
        if integrity_checksums:
            out_sha = _sha256_file(out_path)
            out_sidecar = _write_sha256_sidecar(out_path, out_sha)

        manifest = _base_manifest(args, mode="ab_markdown")
        manifest["spec_count"] = len(specs)
        manifest["only_filter"] = _parse_csv_tokens(args.only)
        manifest["extra_args"] = {
            "base": base_extra_args,
            "a": a_extra,
            "b": b_extra,
        }
        manifest["ab_labels"] = {
            "a": str(args.ab_label_a),
            "b": str(args.ab_label_b),
        }
        manifest["runs"] = run_records
        repro = _repro_context(
            mode="ab_markdown",
            specs=specs,
            extra_args={
                "base": base_extra_args,
                "a": a_extra,
                "b": b_extra,
            },
            args=args,
        )
        manifest["repro_context"] = repro
        manifest["repro_fingerprint_sha256"] = _stable_json_sha256(repro)
        manifest["outputs"] = [{"path": _rel(out_path)}]
        manifest["integrity_checksums_enabled"] = integrity_checksums
        if integrity_checksums and out_sha is not None and out_sidecar is not None:
            manifest["outputs"][0]["sha256"] = out_sha
            manifest["outputs"].append(
                {
                    "path": _rel(out_sidecar),
                    "sha256": _sha256_file(out_sidecar),
                }
            )
        _write_json_file(manifest_path, manifest)
        repro_sidecar = _write_repro_fingerprint_sidecar(
            manifest_path, manifest["repro_fingerprint_sha256"]
        )
        manifest_integrity_sidecar: str | None = None
        if integrity_checksums:
            manifest_sha = _sha256_file(manifest_path)
            manifest_integrity_sidecar = _write_sha256_sidecar(
                manifest_path, manifest_sha
            )

        print(f"Wrote A/B markdown report: {out_path}")
        print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
        print(f"Wrote manifest: {manifest_path}")
        print(f"Wrote reproducibility checksum: {repro_sidecar}")
        if integrity_checksums:
            if out_sidecar is not None:
                print(f"Wrote output integrity checksum: {out_sidecar}")
            if manifest_integrity_sidecar is not None:
                print(f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}")
        return

    if not args.markdown:
        run_records: list[dict[str, Any]] = []
        out_records: list[dict[str, str]] = []
        for spec in specs:
            cmd = spec.cmd + base_extra_args
            raw = _run_and_capture(cmd)
            print(f"Logging to: {spec.txt_out}")
            _write_text_file(spec.txt_out, raw)
            out_sha: str | None = None
            out_sidecar: str | None = None
            if integrity_checksums:
                out_sha = _sha256_file(spec.txt_out)
                out_sidecar = _write_sha256_sidecar(spec.txt_out, out_sha)
            run_records.append(
                {
                    "spec_name": spec.name,
                    "kind": spec.kind,
                    "cmd": cmd,
                    "stdout_sha256": _sha256_text(raw),
                    "output_path": _rel(spec.txt_out),
                }
            )
            out_rec: dict[str, str] = {"path": _rel(spec.txt_out)}
            if integrity_checksums and out_sha is not None and out_sidecar is not None:
                out_rec["sha256"] = out_sha
                run_records[-1]["output_sha256"] = out_sha
                run_records[-1]["output_sha256_file"] = _rel(out_sidecar)
                out_records.append(
                    {
                        "path": _rel(out_sidecar),
                        "sha256": _sha256_file(out_sidecar),
                    }
                )
            out_records.append(out_rec)

        manifest = _base_manifest(args, mode="raw_logs")
        manifest["spec_count"] = len(specs)
        manifest["only_filter"] = _parse_csv_tokens(args.only)
        manifest["extra_args"] = {"base": base_extra_args}
        manifest["runs"] = run_records
        repro = _repro_context(
            mode="raw_logs",
            specs=specs,
            extra_args={"base": base_extra_args},
            args=args,
        )
        manifest["repro_context"] = repro
        manifest["repro_fingerprint_sha256"] = _stable_json_sha256(repro)
        manifest["integrity_checksums_enabled"] = integrity_checksums
        manifest["outputs"] = out_records
        _write_json_file(manifest_path, manifest)
        repro_sidecar = _write_repro_fingerprint_sidecar(
            manifest_path, manifest["repro_fingerprint_sha256"]
        )
        manifest_integrity_sidecar: str | None = None
        if integrity_checksums:
            manifest_sha = _sha256_file(manifest_path)
            manifest_integrity_sidecar = _write_sha256_sidecar(
                manifest_path, manifest_sha
            )
        print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
        print(f"Wrote manifest: {manifest_path}")
        print(f"Wrote reproducibility checksum: {repro_sidecar}")
        if integrity_checksums and manifest_integrity_sidecar is not None:
            print(f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}")
        return

    all_rows: list[tuple[str, int, int, str, str, float, float, float]] = []
    run_records = []
    for spec in specs:
        cmd = spec.cmd + base_extra_args
        raw = _run_and_capture(cmd)
        rows = _parse_rows(raw, spec.kind)
        all_rows.extend(rows)
        run_records.append(
            {
                "spec_name": spec.name,
                "kind": spec.kind,
                "cmd": cmd,
                "stdout_sha256": _sha256_text(raw),
                "parsed_rows": len(rows),
            }
        )

    if len(all_rows) == 0:
        raise RuntimeError(
            "Markdown mode produced no parsed rows; check benchmark output format/regex."
        )

    out_path = os.path.join(REPO_ROOT, args.out)
    _write_text_file(out_path, _to_markdown(all_rows))
    out_sha: str | None = None
    out_sidecar: str | None = None
    if integrity_checksums:
        out_sha = _sha256_file(out_path)
        out_sidecar = _write_sha256_sidecar(out_path, out_sha)

    manifest = _base_manifest(args, mode="markdown")
    manifest["spec_count"] = len(specs)
    manifest["only_filter"] = _parse_csv_tokens(args.only)
    manifest["extra_args"] = {"base": base_extra_args}
    manifest["runs"] = run_records
    repro = _repro_context(
        mode="markdown",
        specs=specs,
        extra_args={"base": base_extra_args},
        args=args,
    )
    manifest["repro_context"] = repro
    manifest["repro_fingerprint_sha256"] = _stable_json_sha256(repro)
    manifest["outputs"] = [{"path": _rel(out_path)}]
    manifest["integrity_checksums_enabled"] = integrity_checksums
    if integrity_checksums and out_sha is not None and out_sidecar is not None:
        manifest["outputs"][0]["sha256"] = out_sha
        manifest["outputs"].append(
            {
                "path": _rel(out_sidecar),
                "sha256": _sha256_file(out_sidecar),
            }
        )
    _write_json_file(manifest_path, manifest)
    repro_sidecar = _write_repro_fingerprint_sidecar(
        manifest_path, manifest["repro_fingerprint_sha256"]
    )
    manifest_integrity_sidecar: str | None = None
    if integrity_checksums:
        manifest_sha = _sha256_file(manifest_path)
        manifest_integrity_sidecar = _write_sha256_sidecar(manifest_path, manifest_sha)

    print(f"Wrote markdown report: {out_path}")
    print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote reproducibility checksum: {repro_sidecar}")
    if integrity_checksums:
        if out_sidecar is not None:
            print(f"Wrote output integrity checksum: {out_sidecar}")
        if manifest_integrity_sidecar is not None:
            print(f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
