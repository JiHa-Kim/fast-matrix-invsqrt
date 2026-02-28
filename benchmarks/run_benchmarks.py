#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix.

Modes:
- default: writes one organized markdown report file (--out)
- --no-markdown: writes per-run .txt logs under
  benchmark_results/runs/<timestamp>_solver_benchmarks/*_solve_logs/
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import shlex
import math
import socket
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

try:
    from .utils import (
        get_git_metadata,
        get_repro_context,
        stable_json_sha256,
        write_text_file,
        write_json_file,
        write_sha256_sidecar,
        format_timestamp,
        repo_relative,
        sha256_file,
        sha256_text,
        write_repro_fingerprint_sidecar,
    )
except ImportError:
    from utils import (
        get_git_metadata,
        get_repro_context,
        stable_json_sha256,
        write_text_file,
        write_json_file,
        write_sha256_sidecar,
        format_timestamp,
        repo_relative,
        sha256_file,
        sha256_text,
        write_repro_fingerprint_sidecar,
    )

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str  # "spd" | "nonspd"
    cmd: list[str]
    txt_out: str


ParsedRow = tuple[
    str, int, int, int, str, str, float, float, float, float, float, float
]


def _assessment_score(row: ParsedRow) -> float:
    rel = float(row[8])
    rel_p90 = float(row[9])
    fail = float(row[10])
    qpm = float(row[11])
    fail_clamped = min(1.0, max(0.0, fail)) if math.isfinite(fail) else 1.0
    if math.isfinite(qpm) and qpm > 0.0:
        base = qpm
    else:
        iter_ms = float(row[7])
        if rel > 0.0 and math.isfinite(rel) and iter_ms > 0.0:
            base = max(0.0, -math.log10(rel)) / iter_ms
        else:
            return float("-inf")
    tail_penalty = 1.0
    if rel > 0.0 and math.isfinite(rel) and math.isfinite(rel_p90) and rel_p90 > 0.0:
        tail_penalty = max(1.0, rel_p90 / rel)
    return (base / tail_penalty) * (1.0 - fail_clamped)


def _run_and_capture(cmd: list[str]) -> str:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT,
    )
    assert result.stdout is not None
    assert result.stderr is not None

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def _drain(
        stream: Any,
        sink: list[str],
        writer: Any,
    ) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            writer.write(line)
            writer.flush()
        stream.close()

    t_out = threading.Thread(
        target=_drain,
        args=(result.stdout, stdout_parts, sys.stdout),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_drain,
        args=(result.stderr, stderr_parts, sys.stderr),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    returncode = result.wait()
    stdout_text = "".join(stdout_parts)
    stderr_text = "".join(stderr_parts)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )
    return stdout_text


def _run_and_write_txt(cmd: list[str], out_path: str) -> None:
    out = _run_and_capture(cmd)
    print(f"Logging to: {out_path}")
    write_text_file(out_path, out)


def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _row_to_dict(row: ParsedRow) -> dict[str, Any]:
    (
        kind,
        p_val,
        n,
        k,
        case_name,
        method,
        total_ms,
        iter_ms,
        relerr,
        relerr_p90,
        failure_rate,
        quality_per_ms,
    ) = row
    return {
        "kind": kind,
        "p": int(p_val),
        "n": int(n),
        "k": int(k),
        "case": case_name,
        "method": method,
        "total_ms": float(total_ms),
        "iter_ms": float(iter_ms),
        "relerr": float(relerr),
        "relerr_p90": float(relerr_p90),
        "failure_rate": float(failure_rate),
        "quality_per_ms": float(quality_per_ms),
    }


def _row_from_dict(obj: Any) -> ParsedRow:
    if not isinstance(obj, dict):
        raise ValueError("row entry must be an object")
    return (
        str(obj["kind"]),
        int(obj["p"]),
        int(obj["n"]),
        int(obj["k"]),
        str(obj["case"]),
        str(obj["method"]),
        float(obj["total_ms"]),
        float(obj["iter_ms"]),
        float(obj["relerr"]),
        float(obj.get("relerr_p90", float("nan"))),
        float(obj.get("failure_rate", float("nan"))),
        float(obj.get("quality_per_ms", float("nan"))),
    )


def _write_rows_cache(path: str, rows: list[ParsedRow]) -> None:
    payload = {
        "schema": "solver_benchmark_rows.v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "rows": [_row_to_dict(r) for r in rows],
    }
    write_json_file(path, payload)


def _load_rows_cache(path: str) -> list[ParsedRow]:
    payload = _load_json_file(path)
    if isinstance(payload, dict):
        rows_raw = payload.get("rows")
    else:
        rows_raw = payload
    if not isinstance(rows_raw, list):
        raise ValueError(f"Invalid rows cache format: expected list at {path}")
    rows: list[ParsedRow] = []
    for obj in rows_raw:
        rows.append(_row_from_dict(obj))
    return rows


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _base_manifest(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "repo_root": repo_relative(REPO_ROOT, REPO_ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv[1:]),
        "args": vars(args),
        "hash_algorithm": "sha256",
        "git": get_git_metadata(REPO_ROOT),
    }


def _repro_context(
    mode: str,
    specs: list[RunSpec],
    extra_args: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    repro_context = get_repro_context(REPO_ROOT, args)
    repro_context.update(
        {
            "mode": mode,
            "spec_count": len(specs),
            "extra_args": extra_args,
            "hostname": socket.gethostname(),
        }
    )
    return repro_context


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

    # Gram RHS (M = G^T B) specialized path compare: primal vs dual
    # Uses the same standard timing/trial defaults as the main suite.
    for p_val in (2, 4):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve_gram_rhs",
            "--p",
            str(p_val),
            "--m",
            "256",
            "--n",
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
                name=f"GRAM RHS p={p_val} m<n",
                kind="spd",
                cmd=cmd,
                txt_out=os.path.join(
                    spd_dir,
                    f"{ts}_gram_rhs_p{p_val}_m256_n1024_k1_16_64.txt",
                ),
            )
        )

    return specs


def _parse_rows(raw: str, kind: str) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    current_p = -1
    current_n = -1
    current_k = -1
    current_case = ""

    hdr_re = re.compile(r"==\s+(?:SPD|Non-SPD)\s+Size\s+(\d+)x\1\s+\|\s+RHS\s+\1x(\d+)")
    p_re = re.compile(r"\bp\s*=\s*(\d+)\b")
    case_re = re.compile(r"^--\s+case\s+([^\s]+)\s+--")
    num = r"(?:[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|inf|nan)"
    line_re = re.compile(
        rf"^(.*?)\s+({num})\s+ms\s+\(pre\s+({num})\s+\+\s+iter\s+({num})\).*?"
        rf"relerr\s+vs\s+(?:true|solve):\s+({num})",
        flags=re.IGNORECASE,
    )
    relerr_p90_re = re.compile(rf"\brelerr_p90\s+({num})", flags=re.IGNORECASE)
    fail_rate_re = re.compile(rf"\bfail_rate\s+({num})%", flags=re.IGNORECASE)
    q_per_ms_re = re.compile(rf"\bq_per_ms\s+({num})", flags=re.IGNORECASE)

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        hm = hdr_re.search(line)
        if hm:
            current_n = int(hm.group(1))
            current_k = int(hm.group(2))
            continue
        pm = p_re.search(line)
        if pm:
            current_p = int(pm.group(1))
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
            p90_m = relerr_p90_re.search(line)
            fail_m = fail_rate_re.search(line)
            qpm_m = q_per_ms_re.search(line)
            relerr_p90 = float(p90_m.group(1)) if p90_m else float("nan")
            failure_rate = float(fail_m.group(1)) / 100.0 if fail_m else float("nan")
            quality_per_ms = float(qpm_m.group(1)) if qpm_m else float("nan")
            rows.append(
                (
                    kind,
                    current_p,
                    current_n,
                    current_k,
                    current_case,
                    method,
                    total_ms,
                    iter_ms,
                    relerr,
                    relerr_p90,
                    failure_rate,
                    quality_per_ms,
                )
            )

    return rows


def _to_markdown(
    all_rows: list[ParsedRow],
    *,
    config: dict[str, Any] | None = None,
) -> str:
    # Heuristic score for identifying winners within _to_markdown
    # (Simplified version of _assessment_score that works directly on ParsedRow)
    def row_score(row: ParsedRow) -> float:
        return _assessment_score(row)

    def clean_method_name(n: str) -> str:
        return (
            n.replace("-Apply", "")
            .replace("Torch-", "T-")
            .replace("-ReuseFactor", "-Reuse")
        )

    out: list[str] = []
    out.append("# Solver Benchmark Report")
    out.append("")
    out.append(f"Generated: {format_timestamp()}")
    out.append("")
    out.append("Assessment metrics:")
    out.append("- `relerr`: median relative error across trials.")
    out.append("- `relerr_p90`: 90th percentile relative error (tail quality).")
    out.append("- `fail_rate`: fraction of failed/non-finite trials.")
    out.append("- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.")
    out.append(
        "- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`."
    )
    out.append("")
    _append_run_config(out, config=config)

    # Group by (kind, p) -> (n, k, case) -> list of rows
    groups = defaultdict(list)
    for row in all_rows:
        kind, p, n, k, case = row[0], row[1], row[2], row[3], row[4]
        groups[(kind, p)].append(row)

    for kind, p in sorted(groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.append(f"## {kind_label} (p={p})")
        out.append("")
        out.append(
            "| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |"
        )
        out.append("|:---|:---|:---|:---|")

        # Group rows by scenario within this section
        scenario_rows = defaultdict(list)
        for row in groups[(kind, p)]:
            n, k, case = row[2], row[3], row[4]
            scenario_rows[(n, k, case)].append(row)

        for n, k, case in sorted(scenario_rows.keys()):
            rows = scenario_rows[(n, k, case)]
            # Find winners
            fastest = min(rows, key=lambda r: r[6])  # total_ms
            accurate = min(rows, key=lambda r: r[8])  # relerr
            best = max(rows, key=row_score)  # score

            scenario_label = f"**{n}** / **{k}**<br>`{case}`"
            f_str = f"{clean_method_name(fastest[5])}<br>({fastest[6]:.2f}ms)"
            a_str = f"{clean_method_name(accurate[5])}<br>({accurate[8]:.1e})"
            w_str = f"**{clean_method_name(best[5])}**"

            out.append(f"| {scenario_label} | {f_str} | {a_str} | {w_str} |")
        out.append("")

    out.append("## Legend")
    out.append("")
    out.append("- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.")
    out.append("- **Fastest**: Method with lowest execution time.")
    out.append("- **Most Accurate**: Method with lowest median relative error.")
    out.append(
        "- **Overall Winner**: Optimal balance of speed and quality (highest assessment score)."
    )
    out.append("")
    out.append("---")
    out.append("")
    out.append("### Detailed Assessment Leaders")
    out.append("")
    out.append(
        "| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |"
    )
    out.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    by_case: dict[tuple[str, int, int, int, str], list[ParsedRow]] = {}
    for row in all_rows:
        key = (row[0], row[1], row[2], row[3], row[4])
        by_case.setdefault(key, []).append(row)
    for key in sorted(by_case.keys()):
        candidates = by_case[key]
        best = max(candidates, key=_assessment_score)
        score = _assessment_score(best)
        out.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {best[5]} | "
            f"{score:.3e} | {best[6]:.3f} | {best[8]:.3e} | {best[9]:.3e} | "
            f"{100.0 * best[10]:.1f}% | {best[11]:.3e} |"
        )
    out.append("")
    return "\n".join(out)


def _append_run_config(
    out: list[str],
    *,
    config: dict[str, Any] | None,
) -> None:
    if not config:
        return
    out.append("Run config:")
    for k in sorted(config.keys()):
        out.append(f"- `{k}`: `{config[k]}`")
    out.append("")


def _to_markdown_ab(
    rows_a: list[ParsedRow],
    rows_b: list[ParsedRow],
    *,
    label_a: str,
    label_b: str,
    match_on_method: bool,
    config: dict[str, Any] | None = None,
) -> str:
    def _key_method(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4], row[5]

    def _key_case(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4]

    def _build_index(
        rows: list[ParsedRow], use_method_key: bool
    ) -> dict[Any, ParsedRow]:
        out: dict[Any, ParsedRow] = {}
        for r in rows:
            key = _key_method(r) if use_method_key else _key_case(r)
            if key in out:
                raise RuntimeError(
                    "A/B compare has duplicate rows per match key. "
                    "Use --methods to keep one method per side, or enable "
                    "--ab-match-on-method when comparing like-for-like methods."
                )
            out[key] = r
        return out

    map_a = _build_index(rows_a, use_method_key=match_on_method)
    map_b = _build_index(rows_b, use_method_key=match_on_method)
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
    out.append("Assessment metrics:")
    out.append("- `relerr`: median relative error across trials.")
    out.append("- `relerr_p90`: 90th percentile relative error (tail quality).")
    out.append("- `fail_rate`: fraction of failed/non-finite trials.")
    out.append("- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.")
    out.append(
        "- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`."
    )
    out.append("")
    _append_run_config(out, config=config)
    out.append(f"A: {label_a}")
    out.append(f"B: {label_b}")
    out.append("")
    if match_on_method:
        out.append(
            "| kind | p | n | k | case | method | "
            f"{label_a}_total_ms | {label_b}_total_ms | delta_ms(B-A) | delta_pct | "
            f"{label_a}_iter_ms | {label_b}_iter_ms | "
            f"{label_a}_relerr | {label_b}_relerr | relerr_ratio(B/A) | "
            f"{label_a}_relerr_p90 | {label_b}_relerr_p90 | "
            f"{label_a}_fail_rate | {label_b}_fail_rate | "
            f"{label_a}_q_per_ms | {label_b}_q_per_ms | q_per_ms_ratio(B/A) |"
        )
        out.append(
            "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
    else:
        out.append(
            "| kind | p | n | k | case | "
            f"{label_a}_method | {label_b}_method | "
            f"{label_a}_total_ms | {label_b}_total_ms | delta_ms(B-A) | delta_pct | "
            f"{label_a}_iter_ms | {label_b}_iter_ms | "
            f"{label_a}_relerr | {label_b}_relerr | relerr_ratio(B/A) | "
            f"{label_a}_relerr_p90 | {label_b}_relerr_p90 | "
            f"{label_a}_fail_rate | {label_b}_fail_rate | "
            f"{label_a}_q_per_ms | {label_b}_q_per_ms | q_per_ms_ratio(B/A) |"
        )
        out.append(
            "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )

    for key in keys:
        ra = map_a[key]
        rb = map_b[key]
        kind, p_val, n, k, case_name = ra[0], ra[1], ra[2], ra[3], ra[4]
        method_a = ra[5]
        method_b = rb[5]
        a_total, a_iter, a_rel = ra[6], ra[7], ra[8]
        b_total, b_iter, b_rel = rb[6], rb[7], rb[8]
        a_rel_p90, a_fail, a_qpm = ra[9], ra[10], ra[11]
        b_rel_p90, b_fail, b_qpm = rb[9], rb[10], rb[11]
        d_ms = b_total - a_total
        d_pct = (100.0 * d_ms / a_total) if a_total != 0 else float("nan")
        rel_ratio = (b_rel / a_rel) if a_rel != 0 else float("nan")
        qpm_ratio = (b_qpm / a_qpm) if a_qpm != 0 else float("nan")
        if match_on_method:
            out.append(
                f"| {kind} | {p_val} | {n} | {k} | {case_name} | {method_a} | "
                f"{a_total:.3f} | {b_total:.3f} | {d_ms:.3f} | {d_pct:.2f}% | "
                f"{a_iter:.3f} | {b_iter:.3f} | {a_rel:.3e} | {b_rel:.3e} | {rel_ratio:.3f} | "
                f"{a_rel_p90:.3e} | {b_rel_p90:.3e} | {100.0 * a_fail:.1f}% | {100.0 * b_fail:.1f}% | "
                f"{a_qpm:.3e} | {b_qpm:.3e} | {qpm_ratio:.3f} |"
            )
        else:
            out.append(
                f"| {kind} | {p_val} | {n} | {k} | {case_name} | {method_a} | {method_b} | "
                f"{a_total:.3f} | {b_total:.3f} | {d_ms:.3f} | {d_pct:.2f}% | "
                f"{a_iter:.3f} | {b_iter:.3f} | {a_rel:.3e} | {b_rel:.3e} | {rel_ratio:.3f} | "
                f"{a_rel_p90:.3e} | {b_rel_p90:.3e} | {100.0 * a_fail:.1f}% | {100.0 * b_fail:.1f}% | "
                f"{a_qpm:.3e} | {b_qpm:.3e} | {qpm_ratio:.3f} |"
            )
    b_faster = 0
    b_better_quality = 0
    b_better_score = 0
    total = len(keys)
    for key in keys:
        ra = map_a[key]
        rb = map_b[key]
        if rb[6] < ra[6]:
            b_faster += 1
        if rb[8] <= ra[8] and rb[9] <= ra[9] and rb[10] <= ra[10]:
            b_better_quality += 1
        if _assessment_score(rb) > _assessment_score(ra):
            b_better_score += 1

    out.append("")
    out.append("## A/B Summary")
    out.append("")
    out.append("| metric | count | share |")
    out.append("|---|---:|---:|")
    out.append(
        f"| B faster (total_ms) | {b_faster} / {total} | "
        f"{(100.0 * b_faster / total) if total > 0 else float('nan'):.1f}% |"
    )
    out.append(
        f"| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | "
        f"{b_better_quality} / {total} | "
        f"{(100.0 * b_better_quality / total) if total > 0 else float('nan'):.1f}% |"
    )
    out.append(
        f"| B better assessment score | {b_better_score} / {total} | "
        f"{(100.0 * b_better_score / total) if total > 0 else float('nan'):.1f}% |"
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maintained solver benchmark suites"
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--timing-reps", type=int, default=10)
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
        "--run-name",
        type=str,
        default="solver_benchmarks",
        help="Custom name for the run directory (prefixed by HHMMSS).",
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
    parser.add_argument(
        "--markdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write one markdown report (default). "
            "Use --no-markdown to write per-run txt logs instead."
        ),
    )
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
        "--ab-match-on-method",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "A/B row matching key policy. "
            "When true (default), rows are matched by kind+p+n+k+case+method. "
            "When false, match by kind+p+n+k+case so A and B may use different methods "
            "(use --methods to keep one method per side)."
        ),
    )
    parser.add_argument(
        "--ab-interleave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In A/B mode without --ab-baseline-rows-in, run A then B per spec "
            "(default) to reduce run-order/thermal bias."
        ),
    )
    parser.add_argument(
        "--ab-out",
        type=str,
        default="",
        help="Output markdown path used for A/B compare mode.",
    )
    parser.add_argument(
        "--ab-baseline-rows-in",
        type=str,
        default="",
        help=(
            "Optional JSON rows cache path used as A-side in A/B mode. "
            "When provided, A-side benchmark commands are skipped."
        ),
    )
    parser.add_argument(
        "--baseline-rows-out",
        type=str,
        default="",
        help=(
            "Optional JSON rows cache path to write parsed rows for future reuse "
            "(for example as --ab-baseline-rows-in in later runs)."
        ),
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

    today_ts = datetime.now().strftime("%Y_%m_%d")
    time_prefix = datetime.now().strftime("%H%M%S")
    run_name = str(args.run_name).strip() or "solver_benchmarks"

    run_dir_rel = os.path.join(
        "benchmark_results", "runs", today_ts, f"{time_prefix}_{run_name}"
    )
    run_dir_abs = os.path.join(REPO_ROOT, run_dir_rel)
    spd_dir_abs = os.path.join(run_dir_abs, "spd_solve_logs")
    nonspd_dir_abs = os.path.join(run_dir_abs, "nonspd_solve_logs")

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
        ts=time_prefix,
    )
    specs = _filter_specs(specs_all, _parse_csv_tokens(args.only))
    if len(specs) == 0:
        raise ValueError("No benchmark specs matched --only filter")

    base_extra_args = _split_extra_args(args.extra_args)
    ab_baseline_rows_in = str(args.ab_baseline_rows_in).strip()
    baseline_rows_out = str(args.baseline_rows_out).strip()
    ab_mode = bool(
        str(args.ab_extra_args_a).strip()
        or str(args.ab_extra_args_b).strip()
        or ab_baseline_rows_in
    )
    integrity_checksums = bool(args.integrity_checksums)

    manifest_path = os.path.join(REPO_ROOT, str(args.manifest_out))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    if ab_mode:
        a_extra = base_extra_args + _split_extra_args(args.ab_extra_args_a)
        b_extra = base_extra_args + _split_extra_args(args.ab_extra_args_b)

        rows_a: list[ParsedRow] = []
        rows_b: list[ParsedRow] = []
        run_records: list[dict[str, Any]] = []

        if ab_baseline_rows_in:
            baseline_in_path = os.path.join(REPO_ROOT, ab_baseline_rows_in)
            rows_a = _load_rows_cache(baseline_in_path)
            run_records.append(
                {
                    "variant": "A",
                    "source": "rows_cache",
                    "rows_cache_path": repo_relative(baseline_in_path, REPO_ROOT),
                    "parsed_rows": len(rows_a),
                }
            )
        elif bool(args.ab_interleave):
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
                        "stdout_sha256": sha256_text(raw_a),
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
                        "stdout_sha256": sha256_text(raw_b),
                        "parsed_rows": len(rows_b_spec),
                    }
                )
        else:
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
                        "stdout_sha256": sha256_text(raw_a),
                        "parsed_rows": len(rows_a_spec),
                    }
                )

            for spec in specs:
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
                        "stdout_sha256": sha256_text(raw_b),
                        "parsed_rows": len(rows_b_spec),
                    }
                )
        if ab_baseline_rows_in:
            for spec in specs:
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
                        "stdout_sha256": sha256_text(raw_b),
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
            match_on_method=bool(args.ab_match_on_method),
            config={
                "trials": int(args.trials),
                "timing_reps": int(args.timing_reps),
                "timing_warmup_reps": int(args.timing_warmup_reps),
                "dtype": str(args.dtype),
                "only": str(args.only),
                "ab_interleave": bool(args.ab_interleave),
                "ab_match_on_method": bool(args.ab_match_on_method),
                "ab_extra_args_a": " ".join(a_extra),
                "ab_extra_args_b": " ".join(b_extra),
            },
        )
        write_text_file(out_path, md_text)
        out_sidecar: str | None = None
        out_sha: str | None = None
        if integrity_checksums:
            out_sha = write_sha256_sidecar(out_path)
            out_sidecar = f"{out_path}.sha256"
        baseline_rows_path_abs: str | None = None
        baseline_rows_sidecar: str | None = None
        if baseline_rows_out:
            baseline_rows_path_abs = os.path.join(REPO_ROOT, baseline_rows_out)
            _write_rows_cache(baseline_rows_path_abs, rows_a)
            if integrity_checksums:
                write_sha256_sidecar(baseline_rows_path_abs)
                baseline_rows_sidecar = f"{baseline_rows_path_abs}.sha256"

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
        manifest["ab_match_on_method"] = bool(args.ab_match_on_method)
        manifest["ab_interleave"] = bool(args.ab_interleave)
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
        manifest["repro_fingerprint_sha256"] = stable_json_sha256(repro)
        manifest["outputs"] = [{"path": repo_relative(out_path, REPO_ROOT)}]
        manifest["integrity_checksums_enabled"] = integrity_checksums
        if integrity_checksums and out_sha is not None and out_sidecar is not None:
            manifest["outputs"][0]["sha256"] = out_sha
            manifest["outputs"].append(
                {
                    "path": repo_relative(out_sidecar, REPO_ROOT),
                    "sha256": sha256_file(out_sidecar),
                }
            )
        if baseline_rows_path_abs is not None:
            row_out_rec: dict[str, Any] = {
                "path": repo_relative(baseline_rows_path_abs, REPO_ROOT)
            }
            if integrity_checksums:
                row_out_rec["sha256"] = sha256_file(baseline_rows_path_abs)
            manifest["outputs"].append(row_out_rec)
            if integrity_checksums and baseline_rows_sidecar is not None:
                manifest["outputs"].append(
                    {
                        "path": repo_relative(baseline_rows_sidecar, REPO_ROOT),
                        "sha256": sha256_file(baseline_rows_sidecar),
                    }
                )
        write_json_file(manifest_path, manifest)
        repro_sidecar = write_repro_fingerprint_sidecar(
            manifest_path, manifest["repro_fingerprint_sha256"]
        )
        manifest_integrity_sidecar: str | None = None
        if integrity_checksums:
            write_sha256_sidecar(manifest_path)
            manifest_integrity_sidecar = f"{manifest_path}.sha256"

        print(f"Wrote A/B markdown report: {out_path}")
        if baseline_rows_path_abs is not None:
            print(f"Wrote baseline rows cache: {baseline_rows_path_abs}")
        print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
        print(f"Wrote manifest: {manifest_path}")
        print(f"Wrote reproducibility checksum: {repro_sidecar}")
        if integrity_checksums:
            if out_sidecar is not None:
                print(f"Wrote output integrity checksum: {out_sidecar}")
            if baseline_rows_sidecar is not None:
                print(
                    f"Wrote baseline rows integrity checksum: {baseline_rows_sidecar}"
                )
            if manifest_integrity_sidecar is not None:
                print(
                    f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}"
                )
        return

    if not args.markdown:
        run_records: list[dict[str, Any]] = []
        out_records: list[dict[str, str]] = []
        all_rows: list[ParsedRow] = []
        for spec in specs:
            cmd = spec.cmd + base_extra_args
            raw = _run_and_capture(cmd)
            rows = _parse_rows(raw, spec.kind)
            all_rows.extend(rows)
            print(f"Logging to: {spec.txt_out}")
            write_text_file(spec.txt_out, raw)
            out_sha: str | None = None
            out_sidecar: str | None = None
            if integrity_checksums:
                out_sha = write_sha256_sidecar(spec.txt_out)
                out_sidecar = f"{spec.txt_out}.sha256"
            run_records.append(
                {
                    "spec_name": spec.name,
                    "kind": spec.kind,
                    "cmd": cmd,
                    "stdout_sha256": sha256_text(raw),
                    "output_path": repo_relative(spec.txt_out, REPO_ROOT),
                    "parsed_rows": len(rows),
                }
            )
            out_rec: dict[str, str] = {"path": repo_relative(spec.txt_out, REPO_ROOT)}
            if integrity_checksums and out_sha is not None and out_sidecar is not None:
                out_rec["sha256"] = out_sha
                run_records[-1]["output_sha256"] = out_sha
                run_records[-1]["output_sha256_file"] = repo_relative(
                    out_sidecar, REPO_ROOT
                )
                out_records.append(
                    {
                        "path": repo_relative(out_sidecar, REPO_ROOT),
                        "sha256": sha256_file(out_sidecar),
                    }
                )
            out_records.append(out_rec)

        baseline_rows_path_abs: str | None = None
        baseline_rows_sidecar: str | None = None
        if baseline_rows_out:
            baseline_rows_path_abs = os.path.join(REPO_ROOT, baseline_rows_out)
            _write_rows_cache(baseline_rows_path_abs, all_rows)
            row_out_rec: dict[str, str] = {
                "path": repo_relative(baseline_rows_path_abs, REPO_ROOT)
            }
            if integrity_checksums:
                row_sha = write_sha256_sidecar(baseline_rows_path_abs)
                row_out_rec["sha256"] = row_sha
                baseline_rows_sidecar = f"{baseline_rows_path_abs}.sha256"
                out_records.append(
                    {
                        "path": repo_relative(baseline_rows_sidecar, REPO_ROOT),
                        "sha256": sha256_file(baseline_rows_sidecar),
                    }
                )
            out_records.append(row_out_rec)

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
        manifest["repro_fingerprint_sha256"] = stable_json_sha256(repro)
        manifest["integrity_checksums_enabled"] = integrity_checksums
        manifest["outputs"] = out_records
        write_json_file(manifest_path, manifest)
        repro_sidecar = write_repro_fingerprint_sidecar(
            manifest_path, manifest["repro_fingerprint_sha256"]
        )
        manifest_integrity_sidecar: str | None = None
        if integrity_checksums:
            write_sha256_sidecar(manifest_path)
            manifest_integrity_sidecar = f"{manifest_path}.sha256"
        print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
        print(f"Wrote manifest: {manifest_path}")
        if baseline_rows_path_abs is not None:
            print(f"Wrote baseline rows cache: {baseline_rows_path_abs}")
        print(f"Wrote reproducibility checksum: {repro_sidecar}")
        if integrity_checksums and manifest_integrity_sidecar is not None:
            print(f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}")
        if integrity_checksums and baseline_rows_sidecar is not None:
            print(f"Wrote baseline rows integrity checksum: {baseline_rows_sidecar}")
        return

    all_rows: list[ParsedRow] = []
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
                "stdout_sha256": sha256_text(raw),
                "parsed_rows": len(rows),
            }
        )

    if len(all_rows) == 0:
        raise RuntimeError(
            "Markdown mode produced no parsed rows; check benchmark output format/regex."
        )

    out_path = os.path.join(REPO_ROOT, args.out)
    write_text_file(
        out_path,
        _to_markdown(
            all_rows,
            config={
                "trials": int(args.trials),
                "timing_reps": int(args.timing_reps),
                "timing_warmup_reps": int(args.timing_warmup_reps),
                "dtype": str(args.dtype),
                "only": str(args.only),
                "extra_args": " ".join(base_extra_args),
            },
        ),
    )
    out_sha: str | None = None
    out_sidecar: str | None = None
    if integrity_checksums:
        out_sha = write_sha256_sidecar(out_path)
        out_sidecar = f"{out_path}.sha256"
    baseline_rows_path_abs: str | None = None
    baseline_rows_sidecar: str | None = None
    if baseline_rows_out:
        baseline_rows_path_abs = os.path.join(REPO_ROOT, baseline_rows_out)
        _write_rows_cache(baseline_rows_path_abs, all_rows)
        if integrity_checksums:
            write_sha256_sidecar(baseline_rows_path_abs)
            baseline_rows_sidecar = f"{baseline_rows_path_abs}.sha256"

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
    manifest["repro_fingerprint_sha256"] = stable_json_sha256(repro)
    manifest["outputs"] = [{"path": repo_relative(out_path, REPO_ROOT)}]
    manifest["integrity_checksums_enabled"] = integrity_checksums
    if integrity_checksums and out_sha is not None and out_sidecar is not None:
        manifest["outputs"][0]["sha256"] = out_sha
        manifest["outputs"].append(
            {
                "path": repo_relative(out_sidecar, REPO_ROOT),
                "sha256": sha256_file(out_sidecar),
            }
        )
    if baseline_rows_path_abs is not None:
        row_out_rec: dict[str, Any] = {
            "path": repo_relative(baseline_rows_path_abs, REPO_ROOT)
        }
        if integrity_checksums:
            row_out_rec["sha256"] = sha256_file(baseline_rows_path_abs)
        manifest["outputs"].append(row_out_rec)
        if integrity_checksums and baseline_rows_sidecar is not None:
            manifest["outputs"].append(
                {
                    "path": repo_relative(baseline_rows_sidecar, REPO_ROOT),
                    "sha256": sha256_file(baseline_rows_sidecar),
                }
            )
    write_json_file(manifest_path, manifest)
    repro_sidecar = write_repro_fingerprint_sidecar(
        manifest_path, manifest["repro_fingerprint_sha256"]
    )
    manifest_integrity_sidecar: str | None = None
    if integrity_checksums:
        write_sha256_sidecar(manifest_path)
        manifest_integrity_sidecar = f"{manifest_path}.sha256"

    print(f"Wrote markdown report: {out_path}")
    if baseline_rows_path_abs is not None:
        print(f"Wrote baseline rows cache: {baseline_rows_path_abs}")
    print(f"Repro fingerprint: {manifest['repro_fingerprint_sha256']}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote reproducibility checksum: {repro_sidecar}")
    if integrity_checksums:
        if out_sidecar is not None:
            print(f"Wrote output integrity checksum: {out_sidecar}")
        if baseline_rows_sidecar is not None:
            print(f"Wrote baseline rows integrity checksum: {baseline_rows_sidecar}")
        if manifest_integrity_sidecar is not None:
            print(f"Wrote manifest integrity checksum: {manifest_integrity_sidecar}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
