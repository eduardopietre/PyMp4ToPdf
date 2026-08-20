import pytest

from mp4_to_pdf_gui import to_per_mile


@pytest.mark.parametrize(
    "num, div, expected",
    [
        (0, 1, 0),
        (0, 100, 0),
        (1, 1, 1000),
        (1, 2, 500),
        (50, 100, 500),
        (100, 100, 1000),
        (3, 10, 300),
        (1, 3, 333),
        (2, 3, 667),
        (999, 1000, 999),
        (1000, 1000, 1000),
        (150, 150, 1000),
        (75, 150, 500),
        (1, 150, 7),
        (149, 150, 993),
    ],
)
def test_to_per_mile_values(num, div, expected):
    assert to_per_mile(num, div) == expected


def test_to_per_mile_matches_progress_convention():
    length = 240
    assert to_per_mile(0, length) == 0
    assert to_per_mile(length, length) == 1000
    assert 0 <= to_per_mile(length // 2, length) <= 1000


def test_to_per_mile_zero_divisor_raises():
    with pytest.raises(ZeroDivisionError):
        to_per_mile(1, 0)


def test_to_per_mile_returns_int():
    assert isinstance(to_per_mile(1, 3), int)
