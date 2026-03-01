#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix and generates Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Any

from benchmarks.utils import (
    get_git_metadata,
    write_text_file,
    write_json_file,
    write_sha256_sidecar,
    format_timestamp,
    repo_relative,
)
from benchmarks.solver_utils import (
    ParsedRow,
    parse_rows,
    row_from_dict,
    row_to_dict,
)
from benchmarks.solver_reporting import to_markdown, to_markdown_ab

# Bootstrap and Common Utils
try:
    from .runner import ensure_repo_root_on_path, run_and_capture, get_run_directory
except ImportError:
    from runner import ensure_repo_root_on_path, run_and_capture, get_run_directory

REPO_ROOT = ensure_repo_root_on_path()


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str  # "spd" | "nonspd"
    cmd: list[str]
    txt_out: str
    tags: dict[str, str]


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
                txt_out=os.path.join(spd_dir, f"{ts}_spd_p{p_val}_klt_n.txt"),
                tags={"p": str(p_val), "n": "1024", "k": "lt_n", "kind": "spd"},
            )
        )

    # SPD, p in {1,2,4}, k=n for n in {256, 512, 1024}
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
                        spd_dir, f"{ts}_spd_p{p_val}_keq_n_{n_val}.txt"
                    ),
                    tags={"p": str(p_val), "n": str(n_val), "k": "eq_n", "kind": "spd"},
                )
            )

    # Non-SPD p=1
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
                txt_out=os.path.join(nonspd_dir, f"{ts}_nonspd_p1_keq_n_{n_val}.txt"),
                tags={"p": "1", "n": str(n_val), "k": "eq_n", "kind": "nonspd"},
            )
        )

    # Gram RHS (M = G^T B)
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
                txt_out=os.path.join(spd_dir, f"{ts}_gram_rhs_p{p_val}.txt"),
                tags={
                    "p": str(p_val),
                    "m": "256",
                    "n": "1024",
                    "k": "lt_n",
                    "kind": "gram",
                },
            )
        )

    return specs


def _parse_csv_tokens(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [tok.strip() for tok in str(spec).split(",") if tok.strip()]


def _filter_specs(
    specs: Iterable[RunSpec],
    only_tokens: list[str],
    kinds: list[str],
    p_vals: list[str],
    sizes: list[str],
) -> list[RunSpec]:
    out = list(specs)
    if only_tokens:
        toks = [t.lower() for t in only_tokens if t]
        out = [
            s for s in out if any(t in f"{s.name} {s.txt_out}".lower() for t in toks)
        ]

    if kinds:
        out = [s for s in out if s.tags.get("kind") in kinds]

    if p_vals:
        out = [s for s in out if s.tags.get("p") in p_vals]

    if sizes:
        out = [s for s in out if s.tags.get("n") in sizes or s.tags.get("m") in sizes]

    return out


def _sanitize_args(args: argparse.Namespace, repo_root: str) -> dict[str, Any]:
    """Convert absolute paths in args to relative ones for cleaner reporting."""
    safe_args = dict(vars(args))
    path_keys = [
        "out",
        "ab_out",
        "manifest_out",
        "ab_baseline_rows_in",
        "baseline_rows_out",
        "best_dir",
        "baseline_dir",
        "golden_dir",
    ]
    for k in path_keys:
        if safe_args.get(k):
            # Use repo_relative from benchmarks.utils
            safe_args[k] = repo_relative(str(safe_args[k]), repo_root)
    return safe_args


STABLE_METHODS = {
    "Torch-Linalg-Solve",
    "Torch-Cholesky-Solve",
    "Torch-EVD-Solve",
    "Torch-Cholesky-Solve-ReuseFactor",
    "Inverse-Newton-Coupled-Apply",
}


def _get_applicable_methods(spec: RunSpec) -> list[str]:
    """Determine the default methods for a spec if not explicitly provided."""
    from benchmarks.solve.bench_solve_core import matrix_solve_methods

    p_val = int(spec.tags.get("p", 1))
    kind = spec.kind
    if kind == "spd":
        return matrix_solve_methods(p_val)
    if kind == "nonspd":
        return [
            "PE-Quad-Coupled-Apply",
            "Inverse-Newton-Coupled-Apply",
            "Torch-Linalg-Solve",
        ]
    if kind == "gram":
        return [
            "PE-Quad-Coupled-Apply-Primal-Gram",
            "PE-Quad-Coupled-Apply-Dual-Gram-RHS",
        ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maintained solver benchmark suites"
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--timing-reps", type=int, default=10)
    parser.add_argument("--timing-warmup-reps", type=int, default=2)
    parser.add_argument(
        "--only", type=str, default="", help="Filter specs by substring."
    )
    parser.add_argument(
        "--kinds", type=str, default="", help="Comma-separated kinds (spd,nonspd,gram)"
    )
    parser.add_argument(
        "--p-vals", type=str, default="", help="Comma-separated p values (1,2,4)"
    )
    parser.add_argument(
        "--sizes", type=str, default="", help="Comma-separated sizes (256,512,1024)"
    )
    parser.add_argument(
        "--methods", type=str, default="", help="Comma-separated methods to run."
    )

    parser.add_argument("--run-name", type=str, default="solver_benchmarks")
    parser.add_argument(
        "--extra-args", type=str, default="", help="Extra args for all commands."
    )
    parser.add_argument(
        "--markdown", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", type=str, default="", help="Output markdown path.")
    parser.add_argument(
        "--prod", action="store_true", help="Update production documentation."
    )
    parser.add_argument("--ab-extra-args-a", type=str, default="")
    parser.add_argument("--ab-extra-args-b", type=str, default="")
    parser.add_argument("--ab-label-a", type=str, default="A")
    parser.add_argument("--ab-label-b", type=str, default="B")
    parser.add_argument(
        "--ab-match-on-method", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ab-interleave", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ab-out", type=str, default="")
    parser.add_argument("--ab-baseline-rows-in", type=str, default="")
    parser.add_argument("--baseline-rows-out", type=str, default="")
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument(
        "--ab-baseline-latest",
        action="store_true",
        help="Use latest run manifest as baseline.",
    )
    parser.add_argument(
        "--integrity-checksums", action=argparse.BooleanOptionalAction, default=True
    )

    # Baseline Results
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare against baseline results.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline results with current run.",
    )
    parser.add_argument(
        "--baseline-dir", type=str, default="benchmark_results/baseline"
    )
    parser.add_argument(
        "--from-baseline",
        action="store_true",
        help="Generate report from baseline results instead of running benchmarks.",
    )

    args = parser.parse_args()

    rel_run_dir, abs_run_dir = get_run_directory(str(args.run_name), REPO_ROOT)
    spd_dir = os.path.join(abs_run_dir, "spd_solve_logs")
    nonspd_dir = os.path.join(abs_run_dir, "nonspd_solve_logs")
    time_prefix = datetime.now().strftime("%H%M%S")

    # Handle baseline results lookup
    if args.baseline and not args.ab_baseline_rows_in:
        baseline_path = os.path.join(
            REPO_ROOT, args.baseline_dir, "baseline_solver.json"
        )
        if os.path.exists(baseline_path):
            args.ab_baseline_rows_in = baseline_path
            args.ab_label_a = "Baseline"
            print(f"[baseline] comparing against: {baseline_path}")

    # Handle latest baseline lookup
    if args.ab_baseline_latest and not args.ab_baseline_rows_in:
        import glob

        all_manifests = glob.glob(
            str(REPO_ROOT / "benchmark_results/runs/**/*run_manifest.json"),
            recursive=True,
        )
        if all_manifests:
            latest_manifest = max(all_manifests, key=os.path.getmtime)
            args.ab_baseline_rows_in = latest_manifest
            print(f"[ab] using latest manifest: {latest_manifest}")

    # Set default paths
    if args.prod:
        args.out = os.path.join(
            REPO_ROOT, "docs", "benchmarks", "benchmark_results_production.md"
        )
    if not args.out:
        args.out = os.path.join(abs_run_dir, "solver_benchmarks.md")
    if not args.ab_out:
        args.ab_out = os.path.join(abs_run_dir, "solver_benchmarks_ab.md")
    if not args.manifest_out:
        args.manifest_out = os.path.join(abs_run_dir, "run_manifest.json")

    specs_all = _build_specs(
        trials=args.trials,
        dtype=args.dtype,
        timing_reps=args.timing_reps,
        warmup_reps=args.timing_warmup_reps,
        spd_dir=spd_dir,
        nonspd_dir=nonspd_dir,
        ts=time_prefix,
    )
    specs = _filter_specs(
        specs_all,
        _parse_csv_tokens(args.only),
        _parse_csv_tokens(args.kinds),
        _parse_csv_tokens(args.p_vals),
        _parse_csv_tokens(args.sizes),
    )

    # Create sanitized args for reporting
    safe_args = _sanitize_args(args, str(REPO_ROOT))

    if args.from_baseline:
        baseline_path = os.path.join(
            REPO_ROOT, args.baseline_dir, "baseline_solver.json"
        )
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(
                f"Cannot use --from-baseline: {baseline_path} not found."
            )

        print(f"[baseline] Generating report from: {baseline_path}")
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_rows = data.get("rows", data.get("results", []))
            all_rows = [row_from_dict(r) for r in raw_rows]

        report = to_markdown(all_rows, config=safe_args)
        write_text_file(args.out, report)
        if args.integrity_checksums:
            write_sha256_sidecar(args.out)
        print(
            f"Done (from baseline). Report at: {repo_relative(args.out, str(REPO_ROOT))}"
        )
        return

    base_extra_args = shlex.split(args.extra_args)
    if args.methods:
        base_extra_args += ["--methods", args.methods]

    ab_mode = bool(
        args.ab_extra_args_a or args.ab_extra_args_b or args.ab_baseline_rows_in
    )

    rows_a: list[ParsedRow] = []
    rows_b: list[ParsedRow] = []
    run_records = []
    all_rows_for_manifest: list[ParsedRow] = []

    # Simplified execution logic
    def run_spec_variant(spec, extra, variant_label):
        cmd = spec.cmd + extra
        raw = run_and_capture(cmd, REPO_ROOT)
        rows = parse_rows(raw, spec.kind)
        write_text_file(spec.txt_out + (f".{variant_label}" if ab_mode else ""), raw)
        return rows, raw, cmd

    if ab_mode:
        a_extra = base_extra_args + shlex.split(args.ab_extra_args_a)
        b_extra = base_extra_args + shlex.split(args.ab_extra_args_b)

        if args.ab_baseline_rows_in:
            with open(args.ab_baseline_rows_in, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_rows = data.get("rows", data.get("results", []))
                rows_a = [row_from_dict(r) for r in raw_rows]
                print(f"[debug] Loaded {len(rows_a)} rows from A baseline.")

        # Build lookup for A rows to allow skipping stable methods
        def _row_key(r: ParsedRow) -> tuple:
            # (kind, p, n, k, case, method)
            return (r[0], r[1], r[2], r[3], r[4], r[5])

        map_a_lookup = {_row_key(r): r for r in rows_a}

        for spec in specs:
            if not args.ab_baseline_rows_in:
                r, raw, cmd = run_spec_variant(spec, a_extra, "A")
                rows_a.extend(r)
                run_records.append(
                    {"spec": spec.name, "variant": "A", "cmd": cmd, "parsed": len(r)}
                )

            # Determine methods to run for B
            current_methods = _get_applicable_methods(spec)
            if args.methods:
                # user requested specific methods on CLI
                to_run = [m.strip() for m in args.methods.split(",") if m.strip()]
            else:
                to_run = []
                for m in current_methods:
                    # Always run non-stable methods (experimental ones)
                    if m not in STABLE_METHODS:
                        to_run.append(m)
                        continue

                    # Heuristic: if we have the method in map_a_lookup for THIS p/n/k, skip.
                    found_in_a = any(
                        rk[1] == int(spec.tags.get("p", -1))
                        and rk[2] == int(spec.tags.get("n", -1))
                        and rk[5] == m
                        for rk in map_a_lookup.keys()
                    )

                    if not found_in_a:
                        to_run.append(m)
                    else:
                        # Copy from A to B so they show up in comparison
                        skipped = [
                            r
                            for r in rows_a
                            if r[5] == m
                            and r[1] == int(spec.tags.get("p", -1))
                            and r[2] == int(spec.tags.get("n", -1))
                        ]
                        rows_b.extend(skipped)
                        all_rows_for_manifest.extend(skipped)

            spec_b_extra = b_extra[:]
            if to_run:
                spec_b_extra += ["--methods", ",".join(to_run)]
                r, raw, cmd = run_spec_variant(spec, spec_b_extra, "B")
                rows_b.extend(r)
                all_rows_for_manifest.extend(r)
                run_records.append(
                    {"spec": spec.name, "variant": "B", "cmd": cmd, "parsed": len(r)}
                )
            else:
                print(f"[skip] {spec.name}: all methods found in baseline.")

        # Filter out stable methods from the A/B report to keep it focused
        filtered_rows_a = [r for r in rows_a if r[5] not in STABLE_METHODS]
        filtered_rows_b = [r for r in rows_b if r[5] not in STABLE_METHODS]

        try:
            report = to_markdown_ab(
                filtered_rows_a,
                filtered_rows_b,
                label_a=args.ab_label_a,
                label_b=args.ab_label_b,
                match_on_method=args.ab_match_on_method,
                config=safe_args,
            )
        except RuntimeError as e:
            print(f"[ab] A/B match failed: {e}")
            report = to_markdown_ab(
                filtered_rows_a,
                filtered_rows_b,
                label_a=args.ab_label_a,
                label_b=args.ab_label_b,
                match_on_method=False,
                config=vars(args),
            )
        write_text_file(args.ab_out, report)

        # Baseline Update Logic
        if args.update_baseline:
            baseline_path = os.path.join(
                REPO_ROOT, args.baseline_dir, "baseline_solver.json"
            )
            print(
                f"[baseline] Promoting current run to baseline results at {baseline_path}"
            )
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            write_json_file(baseline_path, {"rows": [row_to_dict(r) for r in rows_b]})
            write_sha256_sidecar(baseline_path)

    else:
        all_rows = []
        for spec in specs:
            r, raw, cmd = run_spec_variant(spec, base_extra_args, "")
            all_rows.extend(r)
            all_rows_for_manifest.extend(r)
            run_records.append({"spec": spec.name, "cmd": cmd, "parsed": len(r)})

        report = to_markdown(all_rows, config=safe_args)
        write_text_file(args.out, report)

        if args.update_baseline:
            baseline_path = os.path.join(
                REPO_ROOT, args.baseline_dir, "baseline_solver.json"
            )
            print(
                f"[baseline] Promoting current run to baseline results at {baseline_path}"
            )
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            write_json_file(baseline_path, {"rows": [row_to_dict(r) for r in all_rows]})
            write_sha256_sidecar(baseline_path)

    if args.baseline_rows_out:
        write_json_file(
            args.baseline_rows_out,
            {"rows": [row_to_dict(r) for r in (rows_a if ab_mode else all_rows)]},
        )

    manifest = {
        "generated_at": format_timestamp(),
        "args": safe_args,
        "git": get_git_metadata(str(REPO_ROOT)),
        "runs": run_records,
        "rows": [row_to_dict(r) for r in all_rows_for_manifest],
    }
    write_json_file(args.manifest_out, manifest)
    if args.integrity_checksums:
        for p in [args.out, args.ab_out, args.manifest_out]:
            if os.path.exists(p):
                write_sha256_sidecar(p)

    final_report = args.ab_out if ab_mode else args.out
    print(f"Done. Report at: {repo_relative(final_report, str(REPO_ROOT))}")


if __name__ == "__main__":
    main()
