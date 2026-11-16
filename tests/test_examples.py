"""Checks that the maths in stats/examples.py gives the right answers."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from stats.examples import (
    AgreementTable,
    agreement_table,
    bootstrap_ci,
    compare_examples,
    mcnemar_test,
    wilson_interval,
)


def make_answers(both_correct, only_a, only_b, both_wrong):
    """Build two right/wrong lists with the given group sizes."""
    a = [1] * both_correct + [1] * only_a + [0] * only_b + [0] * both_wrong
    b = [1] * both_correct + [0] * only_a + [1] * only_b + [0] * both_wrong
    return a, b


# The worked example: 1,000 test examples, A 83.5% correct, B 85.0% correct.
A, B = make_answers(both_correct=800, only_a=35, only_b=50, both_wrong=115)


# --- Answers we calculated by hand ---------------------------------------


def test_agreement_table_counts_the_four_groups():
    assert agreement_table(A, B) == AgreementTable(800, 35, 50, 115)


def test_accuracies_and_difference():
    result = compare_examples(A, B)
    assert result.accuracy_a == pytest.approx(83.5)
    assert result.accuracy_b == pytest.approx(85.0)
    assert result.difference == pytest.approx(1.5)


def test_mcnemar_on_the_worked_example_is_about_0_13():
    assert mcnemar_test(35, 50) == pytest.approx(0.13, abs=0.01)


def test_mcnemar_small_case_by_hand():
    # 1 versus 5: the chance of 0 or 1 heads in 6 flips is 7/64, doubled.
    assert mcnemar_test(1, 5) == pytest.approx(2 * 7 / 64)


def test_wilson_on_the_worked_example():
    low, high = wilson_interval(835, 1000)
    assert low == pytest.approx(81.1, abs=0.1)
    assert high == pytest.approx(85.7, abs=0.1)


# --- A referee: scipy must agree with ours ---------------------------------


@pytest.mark.parametrize("only_a, only_b", [(35, 50), (1, 5), (0, 9), (20, 20), (300, 250)])
def test_mcnemar_matches_scipy(only_a, only_b):
    referee = scipy_stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    assert mcnemar_test(only_a, only_b) == pytest.approx(referee)


@pytest.mark.parametrize("correct, n", [(835, 1000), (9, 10), (0, 20), (20, 20)])
def test_wilson_matches_scipy(correct, n):
    referee = scipy_stats.binomtest(correct, n).proportion_ci(0.95, method="wilson")
    low, high = wilson_interval(correct, n)
    assert low == pytest.approx(100 * referee.low, abs=1e-6)
    assert high == pytest.approx(100 * referee.high, abs=1e-6)


# --- Sanity checks ---------------------------------------------------------


def test_identical_models_show_no_difference():
    result = compare_examples(A, A)
    assert result.difference == 0
    assert result.p_value == 1
    assert (result.ci_low, result.ci_high) == (0, 0)


def test_mcnemar_is_symmetric():
    assert mcnemar_test(35, 50) == mcnemar_test(50, 35)


def test_wilson_stays_between_0_and_100():
    low, high = wilson_interval(20, 20)
    assert 0 <= low < high <= 100


def test_confidence_interval_contains_the_observed_difference():
    result = compare_examples(A, B)
    assert result.ci_low < result.difference < result.ci_high


def test_same_input_gives_same_answer():
    table = agreement_table(A, B)
    assert bootstrap_ci(table) == bootstrap_ci(table)


def test_row_order_does_not_matter():
    order = np.random.default_rng(0).permutation(len(A))
    shuffled = compare_examples(np.array(A)[order], np.array(B)[order])
    assert shuffled == compare_examples(A, B)


def test_true_false_answers_work_like_1_and_0():
    as_bools = compare_examples([bool(x) for x in A], [bool(x) for x in B])
    assert as_bools == compare_examples(A, B)


def test_a_huge_test_set_is_still_fast_and_valid():
    a, b = make_answers(700_000, 40_000, 42_000, 218_000)
    result = compare_examples(a, b)
    assert 0 < result.p_value < 1e-6
    assert result.ci_low > 0


# --- Bad input gets a clear error ------------------------------------------


def test_different_lengths_are_rejected():
    with pytest.raises(ValueError, match="every test example"):
        compare_examples(A, B[:-1])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no test examples"):
        compare_examples([], [])


def test_values_other_than_0_and_1_are_rejected():
    with pytest.raises(ValueError, match="1 .correct. or 0 .wrong."):
        compare_examples([1, 0, 2], [1, 0, 1])
