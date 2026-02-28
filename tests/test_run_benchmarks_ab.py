import pytest

from benchmarks.run_benchmarks import _parse_rows, _to_markdown_ab


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
