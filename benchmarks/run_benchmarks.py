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
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

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
        default=os.path.join("benchmark_results", "latest_solver_benchmarks.md"),
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
        default=os.path.join("benchmark_results", "latest_solver_benchmarks_ab.md"),
        help="Output markdown path used for A/B compare mode.",
    )
    args = parser.parse_args()

    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")
    if int(args.timing_reps) < 1:
        raise ValueError("--timing-reps must be >= 1")
    if int(args.timing_warmup_reps) < 0:
        raise ValueError("--timing-warmup-reps must be >= 0")

    _ensure_dirs()
    specs_all = _build_specs(
        trials=int(args.trials),
        dtype=str(args.dtype),
        timing_reps=int(args.timing_reps),
        warmup_reps=int(args.timing_warmup_reps),
    )
    specs = _filter_specs(specs_all, _parse_csv_tokens(args.only))
    if len(specs) == 0:
        raise ValueError("No benchmark specs matched --only filter")

    base_extra_args = _split_extra_args(args.extra_args)
    ab_mode = bool(str(args.ab_extra_args_a).strip() or str(args.ab_extra_args_b).strip())

    if ab_mode:
        a_extra = base_extra_args + _split_extra_args(args.ab_extra_args_a)
        b_extra = base_extra_args + _split_extra_args(args.ab_extra_args_b)

        rows_a: list[tuple[str, int, int, str, str, float, float, float]] = []
        rows_b: list[tuple[str, int, int, str, str, float, float, float]] = []

        for spec in specs:
            raw_a = _run_and_capture(spec.cmd + a_extra)
            rows_a.extend(_parse_rows(raw_a, spec.kind))
            raw_b = _run_and_capture(spec.cmd + b_extra)
            rows_b.extend(_parse_rows(raw_b, spec.kind))

        out_path = os.path.join(REPO_ROOT, args.ab_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                _to_markdown_ab(
                    rows_a,
                    rows_b,
                    label_a=str(args.ab_label_a),
                    label_b=str(args.ab_label_b),
                )
            )
        print(f"Wrote A/B markdown report: {out_path}")
        return

    if not args.markdown:
        for spec in specs:
            _run_and_write_txt(spec.cmd + base_extra_args, spec.txt_out)
        return

    all_rows: list[tuple[str, int, int, str, str, float, float, float]] = []
    for spec in specs:
        raw = _run_and_capture(spec.cmd + base_extra_args)
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
