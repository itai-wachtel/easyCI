"""Checks of the planning maths in stats/planning.py."""

import pytest

from stats.planning import fewest_runs_that_can_pass, runs_needed, smallest_possible_p
from stats.seeds import permutation_test

# --- The p-value floor -------------------------------------------------------


def test_floor_for_5_paired_runs_is_2_out_of_32():
    assert smallest_possible_p(5, 5, paired=True) == 2 / 32


def test_floor_for_5_independent_runs_is_2_out_of_252():
    assert smallest_possible_p(5, 5) == 2 / 252


def test_floor_for_groups_of_different_sizes_has_no_mirror_image():
    # 2 and 3 runs: 10 ways to split, and only one is the most extreme.
    assert smallest_possible_p(2, 3) == 1 / 10


@pytest.mark.parametrize(
    "a, b, paired",
    [
        ([1, 2, 3], [10, 11, 12], False),
        ([1, 2], [10, 11, 12], False),
        ([1, 2, 3, 4], [10, 11, 12, 13, 14, 15], False),
        ([1, 2, 3, 4], [11, 13, 15, 17], True),
    ],
)
def test_floor_is_what_the_test_gives_for_the_clearest_possible_result(a, b, paired):
    assert permutation_test(a, b, paired) == pytest.approx(
        smallest_possible_p(len(a), len(b), paired)
    )


def test_fewest_runs_that_can_pass():
    # Paired: 5 runs give 0.0625 (too high), 6 runs give 0.031.
    assert fewest_runs_that_can_pass(paired=True) == 6
    # Independent: 3 runs give 0.1 (too high), 4 runs give 0.029.
    assert fewest_runs_that_can_pass(paired=False) == 4
    # A stricter bar needs more runs.
    assert fewest_runs_that_can_pass(paired=True, confidence=0.99) == 8


# --- Runs needed -------------------------------------------------------------


def test_the_16_over_d_squared_rule():
    assert runs_needed(difference=1.0, noise=1.0) == 16  # d = 1
    assert runs_needed(difference=0.5, noise=1.0) == 63  # d = 0.5
    assert runs_needed(difference=1.0, noise=2.0) == 63  # also d = 0.5


def test_paired_runs_need_half_as_many():
    assert runs_needed(difference=0.5, noise=1.0, paired=True) == 32


def test_direction_of_the_difference_does_not_matter():
    assert runs_needed(-0.5, 1.0) == runs_needed(0.5, 1.0)


def test_smaller_differences_need_more_runs():
    assert runs_needed(0.2, 1.0) > runs_needed(0.4, 1.0) > runs_needed(0.8, 1.0)


def test_stricter_confidence_needs_more_runs():
    assert runs_needed(0.5, 1.0, confidence=0.99) > runs_needed(0.5, 1.0, confidence=0.95)


def test_answer_is_never_below_the_fewest_runs_that_can_pass():
    # A huge effect would need "1 run" by the formula, which cannot pass the bar.
    assert runs_needed(difference=100.0, noise=1.0) == 4
    assert runs_needed(difference=100.0, noise=1.0, paired=True) == 6
    assert runs_needed(difference=1.0, noise=0.0) == 4


def test_a_zero_difference_cannot_be_detected():
    assert runs_needed(difference=0.0, noise=1.0) is None
