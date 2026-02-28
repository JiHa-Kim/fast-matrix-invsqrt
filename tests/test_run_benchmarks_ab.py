import pytest

from benchmarks.run_benchmarks import (
    _parse_rows,
    _row_from_dict,
    _to_markdown,
    _to_markdown_ab,
)


def test_parse_rows_extracts_p_field():
    raw = """
== SPD Size 128x128 | RHS 128x1 | dtype=torch.float32 ==
precond=jacobi | l_target=0.05 | p=2 | ruiz_iters=2
-- case gaussian_spd --
Inverse-Newton-Coupled-Apply      1.234 ms (pre 0.100 + iter 1.134) | relerr vs true: 1.000e-03 | relerr_p90 2.000e-03 | fail_rate 10.0% | q_per_ms 1.234e+00
""".strip()
    rows = _parse_rows(raw, "spd")
    assert len(rows) == 1
    assert rows[0][:6] == (
        "spd",
        2,
        128,
        1,
        "gaussian_spd",
        "Inverse-Newton-Coupled-Apply",
    )
    assert rows[0][8] == pytest.approx(1.0e-3)
    assert rows[0][9] == pytest.approx(2.0e-3)
    assert rows[0][10] == pytest.approx(0.1)
    assert rows[0][11] == pytest.approx(1.234)


def test_parse_rows_back_compat_without_new_metrics():
    raw = """
== SPD Size 64x64 | RHS 64x8 | dtype=torch.float32 ==
precond=jacobi | p=4
-- case illcond_1e6 --
PE-Quad-Coupled-Apply      2.500 ms (pre 0.200 + iter 2.300) | relerr vs true: 9.000e-04
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
    assert rows[0][9] != rows[0][9]  # NaN
    assert rows[0][10] != rows[0][10]  # NaN
    assert rows[0][11] != rows[0][11]  # NaN


def test_parse_rows_keeps_inf_nan_rows():
    raw = """
== Non-SPD Size 64x64 | RHS 64x4 | dtype=torch.float32 ==
p=1
-- case nonnormal_upper --
PE-Quad-Coupled-Apply      inf ms (pre nan + iter inf) | relerr vs solve: inf | relerr_p90 inf | fail_rate 100.0% | q_per_ms nan | bad 5
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
    assert rows[0][10] == pytest.approx(1.0)
    assert rows[0][11] != rows[0][11]  # NaN


def test_ab_compare_can_match_without_method():
    rows_a = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "Inverse-Newton-Coupled-Apply",
            1.0,
            0.9,
            1.0e-3,
            2.0e-3,
            0.0,
            3.0,
        )
    ]
    rows_b = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "PE-Quad-Coupled-Apply",
            0.8,
            0.7,
            1.2e-3,
            1.5e-3,
            0.0,
            3.4,
        )
    ]
    md = _to_markdown_ab(
        rows_a,
        rows_b,
        label_a="baseline",
        label_b="candidate",
        match_on_method=False,
    )
    assert "baseline_method" in md
    assert "candidate_method" in md
    assert "baseline_q_per_ms" in md
    assert "candidate_q_per_ms" in md
    assert "## A/B Summary" in md
    assert "B better assessment score" in md
    assert "Inverse-Newton-Coupled-Apply" in md
    assert "PE-Quad-Coupled-Apply" in md


def test_ab_compare_match_on_method_requires_same_method():
    rows_a = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "Inverse-Newton-Coupled-Apply",
            1.0,
            0.9,
            1.0e-3,
            2.0e-3,
            0.0,
            3.0,
        )
    ]
    rows_b = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "PE-Quad-Coupled-Apply",
            0.8,
            0.7,
            1.2e-3,
            1.5e-3,
            0.0,
            3.4,
        )
    ]
    with pytest.raises(RuntimeError, match="no overlapping keys"):
        _to_markdown_ab(
            rows_a,
            rows_b,
            label_a="baseline",
            label_b="candidate",
            match_on_method=True,
        )


def test_markdown_includes_assessment_leaders():
    rows = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "A",
            1.0,
            0.9,
            1.0e-3,
            1.1e-3,
            0.0,
            3.0,
        ),
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "B",
            0.9,
            0.8,
            1.2e-3,
            1.2e-3,
            0.0,
            3.4,
        ),
    ]
    md = _to_markdown(rows)
    assert "assessment score" in md
    assert "## Assessment Leaders" in md
    assert "| spd | 2 | 128 | 1 | gaussian_spd | B |" in md


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
    assert row[9] != row[9]  # NaN
    assert row[10] != row[10]  # NaN
    assert row[11] != row[11]  # NaN


def test_assessment_leader_penalizes_fail_rate():
    rows = [
        (
            "nonspd",
            1,
            128,
            8,
            "nonnormal_upper",
            "stable",
            1.4,
            1.2,
            2.0e-3,
            2.2e-3,
            0.0,
            2.0,
        ),
        (
            "nonspd",
            1,
            128,
            8,
            "nonnormal_upper",
            "fast_but_unstable",
            1.0,
            0.7,
            1.0e-3,
            1.0e-3,
            0.5,
            4.0,
        ),
    ]
    md = _to_markdown(rows)
    assert "| nonspd | 1 | 128 | 8 | nonnormal_upper | stable |" in md
