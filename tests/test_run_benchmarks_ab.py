import pytest

from benchmarks.solver_utils import parse_rows as _parse_rows, row_from_dict as _row_from_dict
from benchmarks.solver_reporting import to_markdown as _to_markdown, to_markdown_ab as _to_markdown_ab


def test_parse_rows_extracts_p_field():
    raw = """
== SPD Size 128x128 | RHS 128x1 | dtype=torch.float32 ==
precond=jacobi | l_target=0.05 | p=2 | ruiz_iters=2
-- case gaussian_spd --
Inverse-Newton-Coupled-Apply      1.234 ms (pre 0.100 + iter 1.134) | relerr 1.000e-03 (p90 2.000e-03) | fail 10.0% (nf 0.0%, q 10.0%) | q_per_ms 1.234e+00
""".strip()
    rows = _parse_rows(raw, "spd")
    assert len(rows) == 1
    # 0:kind, 1:p, 2:n, 3:k, 4:case, 5:method, 6:total, 7:iter, 8:relerr, 9:p90, 10:nf, 11:qf, 12:fail, 13:qpm, 14:resid, 15:resid_p90
    assert rows[0][:6] == (
        "spd",
        2,
        128,
        1,
        "gaussian_spd",
        "Inverse-Newton-Coupled-Apply",
    )
    assert rows[0][6] == pytest.approx(1.234)
    assert rows[0][7] == pytest.approx(1.134)
    assert rows[0][8] == pytest.approx(1.0e-3)
    assert rows[0][9] == pytest.approx(2.0e-3)
    assert rows[0][10] == pytest.approx(0.0)
    assert rows[0][11] == pytest.approx(0.1)
    assert rows[0][12] == pytest.approx(0.1)
    assert rows[0][13] == pytest.approx(1.234)


def test_parse_rows_back_compat_without_new_metrics():
    raw = """
== SPD Size 64x64 | RHS 64x8 | dtype=torch.float32 ==
precond=jacobi | p=4
-- case illcond_1e6 --
PE-Quad-Coupled-Apply      2.500 ms (pre 0.200 + iter 2.300) | relerr 9.000e-04 (p90 1.000e-03)
""".strip()
    rows = _parse_rows(raw, "spd")
    assert len(rows) == 1
    assert rows[0][:9] == (
        "spd",
        4,
        64,
        8,
        "illcond_1e6",
        "PE-Quad-Coupled-Apply",
        2.5,
        2.3,
        9.0e-4,
    )
    assert rows[0][9] == pytest.approx(1.0e-3)


def test_parse_rows_keeps_inf_nan_rows():
    raw = """
== Non-SPD Size 64x64 | RHS 64x4 | dtype=torch.float32 ==
p=1
-- case nonnormal_upper --
PE-Quad-Coupled-Apply      inf ms (pre nan + iter inf) | relerr inf (p90 inf) | fail 100.0% (nf 0.0%, q 100.0%)
""".strip()
    rows = _parse_rows(raw, "nonspd")
    assert len(rows) == 1
    assert rows[0][:6] == (
        "nonspd",
        1,
        64,
        4,
        "nonnormal_upper",
        "PE-Quad-Coupled-Apply",
    )
    assert rows[0][6] == float("inf")
    assert rows[0][7] == float("inf")
    assert rows[0][8] == float("inf")
    assert rows[0][9] == float("inf")
    assert rows[0][12] == pytest.approx(1.0)


def test_ab_compare_can_match_without_method():
    # Helper to build a valid 16-element row
    def r(kind, p, n, k, case, method, total, iter, relerr, qpm):
        return (kind, p, n, k, case, method, total, iter, relerr, relerr*1.1, 0.0, 0.0, 0.0, qpm, 1e-5, 2e-5)

    rows_a = [r("spd", 2, 128, 1, "gaussian_spd", "Inverse-Newton-Coupled-Apply", 1.0, 0.9, 1.0e-3, 3.0)]
    rows_b = [r("spd", 2, 128, 1, "gaussian_spd", "PE-Quad-Coupled-Apply", 0.8, 0.7, 1.2e-3, 3.4)]
    
    md = _to_markdown_ab(
        rows_a,
        rows_b,
        label_a="baseline",
        label_b="candidate",
        match_on_method=False,
    )
    assert "# SPD" in md
    assert "## p = 2" in md
    assert "#### Case: `gaussian_spd`" in md
    assert "Inverse-Newton-Coupled vs PE-Quad-Coupled" in md
    assert "|  | B |" in md
    assert "| | **ratio** |" in md
    assert "## A/B Summary" in md


def test_ab_compare_match_on_method_requires_same_method():
    def r(kind, p, n, k, case, method):
        return (kind, p, n, k, case, method, 1.0, 0.9, 1e-3, 1.1e-3, 0.0, 0.0, 0.0, 3.0, 1e-5, 2e-5)

    rows_a = [r("spd", 2, 128, 1, "gaussian_spd", "Inverse-Newton-Coupled-Apply")]
    rows_b = [r("spd", 2, 128, 1, "gaussian_spd", "PE-Quad-Coupled-Apply")]
    with pytest.raises(RuntimeError, match="no overlapping keys"):
        _to_markdown_ab(
            rows_a,
            rows_b,
            label_a="baseline",
            label_b="candidate",
            match_on_method=True,
        )


def test_ab_markdown_separator_column_count_matches_header():
    def r(kind, p, n, k, case, method):
        return (kind, p, n, k, case, method, 1.0, 0.9, 1e-3, 1.1e-3, 0.0, 0.0, 0.0, 3.0, 1e-5, 2e-5)

    rows_a = [r("spd", 2, 128, 1, "gaussian_spd", "A")]
    rows_b = [r("spd", 2, 128, 1, "gaussian_spd", "B")]
    md = _to_markdown_ab(
        rows_a,
        rows_b,
        label_a="baseline",
        label_b="candidate",
        match_on_method=False,
    )
    lines = md.splitlines()
    header = next(line for line in lines if line.startswith("| method | side |"))
    sep_idx = lines.index(header) + 1
    sep = lines[sep_idx]
    # | method | side | total_ms | iter_ms | relerr | relerr_p90 | fail_rate |
    assert header.count("|") == sep.count("|")
    assert header.count("|") == 8


def test_markdown_includes_assessment_leaders():
    def r(method, total, relerr, qpm):
        return ("spd", 2, 128, 1, "gaussian_spd", method, total, total*0.9, relerr, relerr*1.1, 0.0, 0.0, 0.0, qpm, 1e-5, 2e-5)

    rows = [
        r("A", 1.0, 1.0e-3, 3.0),
        r("B", 0.9, 1.2e-3, 3.4),
    ]
    md = _to_markdown(rows)
    assert "# SPD" in md
    assert "## p = 2" in md
    assert "### Size 128x128 | RHS 128x1" in md
    assert "#### Case: `gaussian_spd`" in md
    # B is fastest, A has better relerr.
    assert "| B | **0.900** | **0.810** | 1.20e-03 | 1.32e-03 | **0.0%** |" in md


def test_markdown_ab_includes_run_config_when_provided():
    def r(method):
        return ("spd", 2, 64, 1, "gaussian_spd", method, 1.0, 0.8, 1.0e-3, 1.1e-3, 0.0, 0.0, 0.0, 3.0, 1e-5, 2e-5)

    rows_a = [r("A")]
    rows_b = [r("A")]
    md = _to_markdown_ab(
        rows_a,
        rows_b,
        label_a="A",
        label_b="B",
        match_on_method=True,
        config={"trials": 10, "timing_reps": 10},
    )
    assert "## Run Configuration" in md
        
    assert "- trials: `10`" in md
    assert "- timing_reps: `10`" in md


def test_row_from_dict_is_backward_compatible_with_v1_rows():
    row = _row_from_dict(
        {
            "kind": "spd",
            "p": 2,
            "n": 128,
            "k": 1,
            "case": "gaussian_spd",
            "method": "PE-Quad-Coupled-Apply",
            "total_ms": 1.1,
            "iter_ms": 0.9,
            "relerr": 1.0e-3,
        }
    )
    assert row[:9] == (
        "spd",
        2,
        128,
        1,
        "gaussian_spd",
        "PE-Quad-Coupled-Apply",
        1.1,
        0.9,
        1.0e-3,
    )
    # Checks for additional fields (should be NaN)
    assert row[9] != row[9] 
    assert row[12] != row[12]
