"""Standardized Markdown reporting for solver benchmarks."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from .utils import clean_method_name
from .solver_utils import ParsedRow
from .reporting import build_report_header


def _build_legend(ab_mode: bool = False) -> list[str]:
    """Build a standardized legend for benchmark reports."""
    res = [
        "---",
        "",
        "### Legend",
        "- **Bold values** indicate the best performer for that metric in the scenario." if not ab_mode else "- Metrics are compared between Side A and Side B.",
        "- `total_ms`: Total execution time including preprocessing.",
        "- `iter_ms`: Time spent in iterations.",
        "- `relerr`: Median relative error vs ground truth (for SPD) or reference solver (for Non-SPD).",
        "- `relerr_p90`: 90th percentile relative error (tail quality).",
        "- `fail_rate`: Fraction of trials that were non-finite or failed quality checks.",
    ]
    if ab_mode:
        res.extend([
            "- `delta_ms`: Change in total milliseconds (B - A).",
            "- `delta_pct`: Percentage change in total milliseconds relative to A.",
            "- `ratio(B/A)`: Ratio of B's metric to A's metric. < 1.0 means B is smaller/better for errors/time.",
        ])
    res.append("")
    return res


def _group_rows(all_rows: List[ParsedRow]):
    """Group rows by Kind -> p -> (n, k) -> case."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in all_rows:
        kind, p, n, k, case = row[0], row[1], row[2], row[3], row[4]
        groups[kind][p][(n, k)][case].append(row)
    return groups


def to_markdown(
    all_rows: List[ParsedRow],
    *,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate a full solver benchmark markdown report."""

    def internal_clean(n: str) -> str:
        return clean_method_name(n).replace("-Reuse", "-R")

    out: list[str] = build_report_header("Solver Benchmark Report", config or {})
    out.extend(_build_legend(ab_mode=False))

    # Hierarchy: Kind -> p -> (n, k) -> case -> Methods
    kind_groups = _group_rows(all_rows)

    for kind in sorted(kind_groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.append(f"# {kind_label}")
        out.append("")

        for p in sorted(kind_groups[kind].keys()):
            out.append(f"## p = {p}")
            out.append("")

            for (n, k) in sorted(kind_groups[kind][p].keys()):
                out.append(f"### Size {n}x{n} | RHS {n}x{k}")
                out.append("")

                for case in sorted(kind_groups[kind][p][(n, k)].keys()):
                    out.append(f"#### Case: `{case}`")
                    out.append("")
                    out.append(
                        "| method | total_ms | iter_ms | relerr | relerr_p90 | fail_rate |"
                    )
                    out.append("|:---|---:|---:|---:|---:|---:|")

                    rows = kind_groups[kind][p][(n, k)][case]

                    # Find bests for this block
                    best_total = min(r[6] for r in rows)
                    best_iter = min(r[7] for r in rows)
                    best_relerr = min(r[8] for r in rows)
                    best_relerr_p90 = min(r[9] for r in rows)
                    best_fail = min(r[12] for r in rows)

                    def fmt(val, best, s, is_fail=False):
                        if is_fail and val >= 1.0:
                            return s
                        is_best = (val <= best)
                        if is_best and not math.isnan(val):
                            return f"**{s}**"
                        return s

                    for row in sorted(rows, key=lambda r: r[6]):
                        method = internal_clean(row[5])
                        total_ms = row[6]
                        iter_ms = row[7]
                        relerr = row[8]
                        relerr_p90 = row[9]
                        fail_rate = row[12]

                        s_total = fmt(total_ms, best_total, f"{total_ms:.3f}")
                        s_iter = fmt(iter_ms, best_iter, f"{iter_ms:.3f}")
                        s_rel = fmt(relerr, best_relerr, f"{relerr:.2e}")
                        s_rel_p90 = fmt(relerr_p90, best_relerr_p90, f"{relerr_p90:.2e}")
                        s_fail = fmt(fail_rate, best_fail, f"{100.0*fail_rate:.1f}%", is_fail=True)

                        out.append(
                            f"| {method} | {s_total} | {s_iter} | {s_rel} | {s_rel_p90} | "
                            f"{s_fail} |"
                        )
                    out.append("")
    
    return "\n".join(out)


def to_markdown_ab(
    rows_a: List[ParsedRow],
    rows_b: List[ParsedRow],
    *,
    label_a: str,
    label_b: str,
    match_on_method: bool,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate an A/B solver benchmark markdown comparison."""

    def internal_clean(n: str) -> str:
        return clean_method_name(n).replace("-Reuse", "-R")

    def _key_method(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4], row[5]

    def _key_case(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4]

    def _build_index(
        rows: list[ParsedRow], use_method_key: bool
    ) -> dict[Any, ParsedRow]:
        out_idx: dict[Any, ParsedRow] = {}
        for r in rows:
            key = _key_method(r) if use_method_key else _key_case(r)
            if key in out_idx:
                raise RuntimeError(
                    f"A/B compare has duplicate rows per match key: {key}. "
                    "Use --methods to keep one method per side, or enable "
                    "--ab-match-on-method when comparing like-for-like methods."
                )
            out_idx[key] = r
        return out_idx

    map_a = _build_index(rows_a, use_method_key=match_on_method)
    map_b = _build_index(rows_b, use_method_key=match_on_method)
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(keys) == 0:
        raise RuntimeError(
            "A/B rows had no overlapping keys; cannot build comparable report."
        )

    out: list[str] = build_report_header("Solver Benchmark A/B Report", config or {})
    out.extend(_build_legend(ab_mode=True))

    out.extend(
        [
            f"A: {label_a}",
            f"B: {label_b}",
            "",
        ]
    )

    # Group matched pairs by Kind -> p -> (n, k) -> case
    kind_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    b_faster = 0
    b_better_quality = 0

    for key in keys:
        ra = map_a[key]
        rb = map_b[key]
        kind, p, n, k, case = ra[0], ra[1], ra[2], ra[3], ra[4]
        kind_groups[kind][p][(n, k)][case].append((ra, rb))

        if rb[6] < ra[6]:
            b_faster += 1
        # relerr (8), relerr_p90 (9), fail_rate (12)
        if rb[8] <= ra[8] and rb[9] <= ra[9] and rb[12] <= ra[12]:
            b_better_quality += 1

    for kind in sorted(kind_groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.append(f"# {kind_label}")
        out.append("")

        for p in sorted(kind_groups[kind].keys()):
            out.append(f"## p = {p}")
            out.append("")

            for (n, k) in sorted(kind_groups[kind][p].keys()):
                out.append(f"### Size {n}x{n} | RHS {n}x{k}")
                out.append("")

                for case in sorted(kind_groups[kind][p][(n, k)].keys()):
                    out.append(f"#### Case: `{case}`")
                    out.append("")
                    out.append(
                        "| method | side | total_ms | iter_ms | relerr | relerr_p90 | fail_rate |"
                    )
                    out.append("|:---|:---|---:|---:|---:|---:|---:|")

                    pairs = kind_groups[kind][p][(n, k)][case]

                    def fmt_ab(va, vb, s, is_fail=False):
                        if is_fail and vb >= 1.0:
                            return s
                        if math.isnan(va) or math.isnan(vb):
                            return s
                        better_b = (vb < va)
                        if better_b:
                            return f"**{s}**"
                        return s

                    for ra, rb in sorted(pairs, key=lambda pair: pair[0][6]):
                        method_a = internal_clean(ra[5])
                        method_b = internal_clean(rb[5])
                        
                        m_label = method_a if match_on_method else f"{method_a} vs {method_b}"
                        
                        # Row A
                        out.append(
                            f"| {m_label} | A | {ra[6]:.3f} | {ra[7]:.3f} | {ra[8]:.2e} | {ra[9]:.2e} | "
                            f"{100.0*ra[12]:.1f}% |"
                        )
                        
                        # Row B with bolding if better than A
                        s_total = fmt_ab(ra[6], rb[6], f"{rb[6]:.3f}")
                        s_iter = fmt_ab(ra[7], rb[7], f"{rb[7]:.3f}")
                        s_rel = fmt_ab(ra[8], rb[8], f"{rb[8]:.2e}")
                        s_rel_p90 = fmt_ab(ra[9], rb[9], f"{rb[9]:.2e}")
                        s_fail = fmt_ab(ra[12], rb[12], f"{100.0*rb[12]:.1f}%", is_fail=True)

                        out.append(
                            f"| | B | {s_total} | {s_iter} | {s_rel} | {s_rel_p90} | "
                            f"{s_fail} |"
                        )
                        
                        # Diff/Ratio Row
                        d_ms = rb[6] - ra[6]
                        d_pct = (100.0 * d_ms / ra[6]) if ra[6] != 0 else float("nan")
                        rel_ratio = (rb[8] / ra[8]) if ra[8] != 0 else float("nan")
                        
                        s_d_ms = f"{d_ms:+.3f}"
                        s_d_pct = f"({d_pct:+.1f}%)"
                        s_rel_ratio = f"{rel_ratio:.2f}x"
                        
                        out.append(
                            f"| | **ratio** | {s_d_ms} | {s_d_pct} | {s_rel_ratio} | | |"
                        )
                    out.append("")

    total = len(keys)
    out.append("")
    out.append("## A/B Summary")
    out.append("")
    out.append("| metric | count | share |")
    out.append("|---|---:|---:|")
    out.append(
        f"| B faster (total_ms) | {b_faster} / {total} | "
        f"{(100.0 * b_faster / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append(
        f"| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | "
        f"{b_better_quality} / {total} | "
        f"{(100.0 * b_better_quality / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append("")
    return "\n".join(out)
