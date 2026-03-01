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


def _clean_method(n: str) -> str:
    """Clean and shorten method names for report display."""
    return clean_method_name(n).replace("-Reuse", "-R")


def _format_row_cells(row: ParsedRow, bests: dict[int, float] | None = None) -> list[str]:
    """Format cells for a single solver result row."""
    # Indexes: 6:total, 7:iter, 8:relerr, 9:p90, 12:fail
    def fmt(idx: int, val: float, s: str, is_fail: bool = False):
        if is_fail and val >= 1.0:
            return s
        if bests and idx in bests and not math.isnan(val) and val <= bests[idx]:
            return f"**{s}**"
        return s

    return [
        fmt(6, row[6], f"{row[6]:.3f}"),
        fmt(7, row[7], f"{row[7]:.3f}"),
        fmt(8, row[8], f"{row[8]:.2e}"),
        fmt(9, row[9], f"{row[9]:.2e}"),
        fmt(12, row[12], f"{100.0*row[12]:.1f}%", is_fail=True),
    ]


def to_markdown(
    all_rows: List[ParsedRow],
    *,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate a full solver benchmark markdown report."""
    out: list[str] = build_report_header("Solver Benchmark Report", config or {})
    out.extend(_build_legend(ab_mode=False))

    # Hierarchy: Kind -> p -> (n, k) -> case -> Methods
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in all_rows:
        groups[r[0]][r[1]][(r[2], r[3])][r[4]].append(r)

    for kind in sorted(groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.extend([f"# {kind_label}", ""])

        for p in sorted(groups[kind].keys()):
            out.extend([f"## p = {p}", ""])

            for (n, k) in sorted(groups[kind][p].keys()):
                out.extend([f"### Size {n}x{n} | RHS {n}x{k}", ""])

                for case, rows in sorted(groups[kind][p][(n, k)].items()):
                    out.extend([f"#### Case: `{case}`", ""])
                    out.append("| method | total_ms | iter_ms | relerr | relerr_p90 | fail_rate |")
                    out.append("|:---|---:|---:|---:|---:|---:|")

                    bests = {
                        6: min(r[6] for r in rows),
                        7: min(r[7] for r in rows),
                        8: min(r[8] for r in rows),
                        9: min(r[9] for r in rows),
                        12: min(r[12] for r in rows),
                    }

                    for r in sorted(rows, key=lambda x: x[6]):
                        cells = [_clean_method(r[5])] + _format_row_cells(r, bests)
                        out.append("| " + " | ".join(cells) + " |")
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
    def _key(r: ParsedRow):
        return (r[0], r[1], r[2], r[3], r[4], r[5]) if match_on_method else (r[0], r[1], r[2], r[3], r[4])

    map_a = { _key(r): r for r in rows_a }
    map_b = { _key(r): r for r in rows_b }
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))

    if not keys:
        raise RuntimeError("A/B rows had no overlapping keys; cannot build report.")

    out: list[str] = build_report_header("Solver Benchmark A/B Report", config or {})
    out.extend(_build_legend(ab_mode=True))
    out.extend([f"A: {label_a}", f"B: {label_b}", ""])

    # Hierarchy: Kind -> p -> (n, k) -> case -> list of (ra, rb)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    b_faster = 0
    b_better_q = 0

    for k in keys:
        ra, rb = map_a[k], map_b[k]
        groups[ra[0]][ra[1]][(ra[2], ra[3])][ra[4]].append((ra, rb))
        if rb[6] < ra[6]: b_faster += 1
        if rb[8] <= ra[8] and rb[9] <= ra[9] and rb[12] <= ra[12]: b_better_q += 1

    for kind in sorted(groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.extend([f"# {kind_label}", ""])

        for p in sorted(groups[kind].keys()):
            out.extend([f"## p = {p}", ""])

            for (n, k_rhs) in sorted(groups[kind][p].keys()):
                out.extend([f"### Size {n}x{n} | RHS {n}x{k_rhs}", ""])

                for case, pairs in sorted(groups[kind][p][(n, k_rhs)].items()):
                    out.extend([f"#### Case: `{case}`", ""])
                    out.append("| method | side | total_ms | iter_ms | relerr | relerr_p90 | fail_rate |")
                    out.append("|:---|:---|---:|---:|---:|---:|---:|")

                    for ra, rb in sorted(pairs, key=lambda x: x[0][6]):
                        m_label = _clean_method(ra[5]) if match_on_method else f"{_clean_method(ra[5])} vs {_clean_method(rb[5])}"
                        
                        # Row A
                        cells_a = [m_label, "A"] + _format_row_cells(ra)
                        out.append("| " + " | ".join(cells_a) + " |")

                        # Row B (bold if better than A)
                        bests_b = { i: ra[i] for i in [6, 7, 8, 9, 12] }
                        cells_b = ["", "B"] + _format_row_cells(rb, bests_b)
                        out.append("| " + " | ".join(cells_b) + " |")

                        # Ratio Row
                        d_ms = rb[6] - ra[6]
                        d_pct = (100.0 * d_ms / ra[6]) if ra[6] != 0 else float("nan")
                        rel_ratio = (rb[8] / ra[8]) if ra[8] != 0 else float("nan")
                        out.append(f"| | **ratio** | {d_ms:+.3f} | ({d_pct:+.1f}%) | {rel_ratio:.2f}x | | |")
                    out.append("")

    total = len(keys)
    out.extend(["## A/B Summary", "", "| metric | count | share |", "|---|---:|---:|"])
    out.append(f"| B faster (total_ms) | {b_faster} / {total} | {(100.0*b_faster/total) if total else 0:.1f}% |")
    out.append(f"| B better-or-equal quality | {b_better_q} / {total} | {(100.0*b_better_q/total) if total else 0:.1f}% |")
    out.append("")

    return "\n".join(out)
