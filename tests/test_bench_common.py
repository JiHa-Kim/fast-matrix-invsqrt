import math

from benchmarks.common import median


def test_median_empty_is_nan():
    assert math.isnan(median([]))


def test_median_odd_count():
    assert median([3.0, 1.0, 2.0]) == 2.0


def test_median_even_count_uses_midpoint():
    assert median([1.0, 100.0]) == 50.5
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


