#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix.

Modes:
- default: writes per-run .txt logs under benchmark_results/latest_*_solve_logs/
- --markdown: writes one organized markdown report file (--out)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark_results")
SPD_DIR = os.path.join(RESULTS_DIR, "latest_spd_solve_logs")
NONSPD_DIR = os.path.join(RESULTS_DIR, "latest_nonspd_solve_logs")


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str  # "spd" | "nonspd"
    cmd: list[str]
    txt_out: str


def _ensure_dirs() -> None:
    os.makedirs(SPD_DIR, exist_ok=True)
    os.makedirs(NONSPD_DIR, exist_ok=True)


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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)


def _build_specs(trials: int, dtype: str, timing_reps: int, warmup_reps: int) -> list[RunSpec]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    specs: list[RunSpec] = []

    # SPD, p in {1,2,4}, k<n (n={1024,2048}, k={1,16,64})
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
            "1024,2048",
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
                    SPD_DIR,
                    f"spd_p{p_val}_klt_n_sizes1024_2048_k1_16_64_{ts}.txt",
                ),
            )
        )

    # SPD, p in {1,2,4}, k=n for n in {256,512,1024,2048}
    for p_val in (1, 2, 4):
        for n_val in (256, 512, 1024, 2048):
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
                        SPD_DIR,
                        f"spd_p{p_val}_keq_n_n{n_val}_k{n_val}_{ts}.txt",
                    ),
                )
            )

    # non-SPD p=1, k<n
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "benchmarks.solve.matrix_solve_nonspd",
        "--p",
        "1",
        "--sizes",
        "1024,2048",
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
                NONSPD_DIR,
                f"nonspd_p1_klt_n_sizes1024_2048_k1_16_64_{ts}.txt",
            ),
        )
    )

    # non-SPD p=1, k=n
    for n_val in (256, 512, 1024, 2048):
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
                    NONSPD_DIR,
                    f"nonspd_p1_keq_n_n{n_val}_k{n_val}_{ts}.txt",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maintained solver benchmark suites")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--timing-reps", type=int, default=5)
    parser.add_argument("--timing-warmup-reps", type=int, default=2)
    parser.add_argument("--markdown", action="store_true", help="Write one markdown report instead of per-run txt files")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("benchmark_results", "latest_solver_benchmarks.md"),
        help="Output markdown path used with --markdown",
    )
    args = parser.parse_args()

    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")
    if int(args.timing_reps) < 1:
        raise ValueError("--timing-reps must be >= 1")
    if int(args.timing_warmup_reps) < 0:
        raise ValueError("--timing-warmup-reps must be >= 0")

    _ensure_dirs()
    specs = _build_specs(
        trials=int(args.trials),
        dtype=str(args.dtype),
        timing_reps=int(args.timing_reps),
        warmup_reps=int(args.timing_warmup_reps),
    )

    if not args.markdown:
        for spec in specs:
            _run_and_write_txt(spec.cmd, spec.txt_out)
        return

    all_rows: list[tuple[str, int, int, str, str, float, float, float]] = []
    for spec in specs:
        raw = _run_and_capture(spec.cmd)
        rows = _parse_rows(raw, spec.kind)
        all_rows.extend(rows)

    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown(all_rows))
    print(f"Wrote markdown report: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
