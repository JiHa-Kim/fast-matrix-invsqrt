"""Standardized Markdown reporting for solver benchmarks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .utils import clean_method_name
from .solver_utils import ParsedRow, assessment_score
from .reporting import build_report_header


def to_markdown(
    all_rows: List[ParsedRow],
    *,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate a full solver benchmark markdown report."""

    def internal_clean(n: str) -> str:
        return clean_method_name(n).replace("-Reuse", "-R")

    out: list[str] = build_report_header("Solver Benchmark Report", config or {})

    out.extend(
        [
            "Assessment metrics:",
            "- `relerr`: median relative error across trials.",
            "- `relerr_p90`: 90th percentile relative error (tail quality).",
            "- `fail_rate`: fraction of failed/non-finite trials.",
            "- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.",
            "- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.",
            "",
        ]
    )

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
            best = max(rows, key=assessment_score)  # score

            scenario_label = f"**{n}** / **{k}**<br>`{case}`"
            f_str = f"{internal_clean(fastest[5])}<br>({fastest[6]:.2f}ms)"
            a_str = f"{internal_clean(accurate[5])}<br>({accurate[8]:.1e})"
            w_str = f"**{internal_clean(best[5])}**"

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
        "| kind | p | n | k | case | best_method | score | total_ms | relerr | resid | relerr_p90 | fail_rate | q_per_ms |"
    )
    out.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")

    by_case: dict[tuple[str, int, int, int, str], list[ParsedRow]] = {}
    for row in all_rows:
        key = (row[0], row[1], row[2], row[3], row[4])
        by_case.setdefault(key, []).append(row)

    for key in sorted(by_case.keys()):
        candidates = by_case[key]
        best = max(candidates, key=assessment_score)
        score = assessment_score(best)
        out.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {best[5]} | "
            f"{score:.3e} | {best[6]:.3f} | {best[8]:.3e} | {best[12]:.3e} | {best[9]:.3e} | "
            f"{100.0 * best[10]:.1f}% | {best[11]:.3e} |"
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

    out.extend(
        [
            "Assessment metrics:",
            "- `relerr`: median relative error across trials.",
            "- `relerr_p90`: 90th percentile relative error (tail quality).",
            "- `fail_rate`: fraction of failed/non-finite trials.",
            "- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.",
            "- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.",
            "",
            f"A: {label_a}",
            f"B: {label_b}",
            "",
        ]
    )

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

    b_faster = 0
    b_better_quality = 0
    b_better_score = 0

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

        if rb[6] < ra[6]:
            b_faster += 1
        if rb[8] <= ra[8] and rb[9] <= ra[9] and rb[10] <= ra[10]:
            b_better_quality += 1
        if assessment_score(rb) > assessment_score(ra):
            b_better_score += 1

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
    out.append(
        f"| B better assessment score | {b_better_score} / {total} | "
        f"{(100.0 * b_better_score / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append("")
    return "\n".join(out)
