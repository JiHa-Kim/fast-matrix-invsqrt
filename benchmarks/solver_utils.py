"""Utilities for parsing and assessing solver benchmark results."""

from __future__ import annotations

import re
from typing import Any, List, Tuple

# Type alias for benchmark rows
# (kind, p, n, k, case, method, total_ms, iter_ms, relerr, relerr_p90, nf_rate, qf_rate, fail_rate, qpm, resid, resid_p90)
ParsedRow = Tuple[
    str,
    int,
    int,
    int,
    str,
    str,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def parse_rows(raw: str, kind: str) -> List[ParsedRow]:
    """Parse benchmark stdout into structured rows."""
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
        rf"relerr\s+({num})\s+\(p90\s+({num})\)",
        flags=re.IGNORECASE,
    )
    resid_re = re.compile(
        rf"\bresid\s+({num})\s+\(p90\s+({num})\)", flags=re.IGNORECASE
    )
    fail_rate_re = re.compile(
        rf"\bfail\s+({num})%\s+\(nf\s+({num})%,\s+q\s+({num})%\)", flags=re.IGNORECASE
    )
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
            relerr_p90 = float(lm.group(6))

            resid_m = resid_re.search(line)
            fail_m = fail_rate_re.search(line)
            qpm_m = q_per_ms_re.search(line)

            resid = float(resid_m.group(1)) if resid_m else float("nan")
            resid_p90 = float(resid_m.group(2)) if resid_m else float("nan")

            failure_rate = float(fail_m.group(1)) / 100.0 if fail_m else float("nan")
            nf_rate = float(fail_m.group(2)) / 100.0 if fail_m else float("nan")
            qf_rate = float(fail_m.group(3)) / 100.0 if fail_m else float("nan")

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
                    nf_rate,
                    qf_rate,
                    failure_rate,
                    quality_per_ms,
                    resid,
                    resid_p90,
                )
            )

    return rows


def row_to_dict(row: ParsedRow) -> dict[str, Any]:
    """Convert a ParsedRow tuple to a dictionary for JSON serialization."""
    return {
        "kind": row[0],
        "p": int(row[1]),
        "n": int(row[2]),
        "k": int(row[3]),
        "case": row[4],
        "method": row[5],
        "total_ms": float(row[6]),
        "iter_ms": float(row[7]),
        "relerr": float(row[8]),
        "relerr_p90": float(row[9]),
        "nf_rate": float(row[10]),
        "qf_rate": float(row[11]),
        "failure_rate": float(row[12]),
        "quality_per_ms": float(row[13]),
        "residual": float(row[14]),
        "residual_p90": float(row[15]),
    }


def row_from_dict(obj: dict[str, Any]) -> ParsedRow:
    """Convert a dictionary back to a ParsedRow tuple."""
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
        float(obj.get("nf_rate", float("nan"))),
        float(obj.get("qf_rate", float("nan"))),
        float(obj.get("failure_rate", float("nan"))),
        float(obj.get("quality_per_ms", float("nan"))),
        float(obj.get("residual", float("nan"))),
        float(obj.get("residual_p90", float("nan"))),
    )
