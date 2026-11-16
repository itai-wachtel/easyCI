"""Checks that the maths in stats/seeds.py gives the right answers."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from stats.seeds import (
    bootstrap_ci,
    t_interval,
    cohens_d,
    compare,
    permutation_test,
    probability_of_superiority,
    summarize,
)

# Example 1: independent runs (unpaired).
A = [83.1, 83.9, 83.5, 84.0, 83.2]
B = [84.2, 83.8, 84.5, 84.1, 84.4]

# Example 2: both models ran on the same five seeds (paired).
PAIRED_A = [80.0, 85.0, 82.0, 88.0, 84.0]
PAIRED_B = [80.6, 85.5, 82.7, 88.4, 84.8]


# --- Answers we calculated by hand ---------------------------------------


def test_summary_matches_hand_calculation():
    summary = summarize(A)
    assert summary.mean == pytest.approx(83.54)
    assert summary.std == pytest.approx(0.404, abs=0.001)
    assert summary.n == 5


def test_unpaired_p_value_is_8_out_of_252():
    assert permutation_test(A, B) == pytest.approx(8 / 252)


def test_paired_p_value_is_2_out_of_32():
    assert permutation_test(PAIRED_A, PAIRED_B, paired=True) == pytest.approx(2 / 32)


def test_cohens_d_unpaired():
    assert cohens_d(A, B) == pytest.approx(1.913, abs=0.001)


def test_cohens_d_paired():
    # mean difference 0.6, spread of the differences 0.158
    assert cohens_d(PAIRED_A, PAIRED_B, paired=True) == pytest.approx(3.795, abs=0.001)


def test_superiority_unpaired_is_23_of_25():
    assert probability_of_superiority(A, B) == pytest.approx(23 / 25)


def test_superiority_paired_b_wins_every_seed():
    assert probability_of_superiority(PAIRED_A, PAIRED_B, paired=True) == 1.0


def test_superiority_when_lower_is_better():
    # If these were error rates, B (the higher one) would be the worse model.
    assert probability_of_superiority(A, B, higher_is_better=False) == pytest.approx(2 / 25)


# --- A referee: scipy's own permutation test must agree with ours --------


def mean_difference(x, y):
    return np.mean(y) - np.mean(x)


# Two models with different amounts of noise and different numbers of runs:
# the case where the plain and the studentized test disagree.
UNEVEN_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
UNEVEN_B = [-12.0, 30.0, 9.0, 21.0]


def welch_t(x, y):
    """The difference divided by its standard error, as scipy computes it."""
    return scipy_stats.ttest_ind(y, x, equal_var=False).statistic


# easyCI counts the shuffles whose gap is at least as far from zero as the
# real one. scipy is asked for exactly that count: "how often is the size of
# the gap greater or equal". (scipy's own two-sided default doubles the
# smaller tail instead, which differs when the groups have different sizes.)


def test_plain_p_value_matches_scipy():
    for a, b in [(A, B), (UNEVEN_A, UNEVEN_B)]:
        referee = scipy_stats.permutation_test(
            (a, b), lambda x, y: abs(mean_difference(x, y)), alternative="greater",
            permutation_type="independent", n_resamples=np.inf,
        )
        assert permutation_test(a, b, studentized=False) == pytest.approx(referee.pvalue)


def test_studentized_p_value_matches_scipy():
    for a, b in [(A, B), (UNEVEN_A, UNEVEN_B)]:
        referee = scipy_stats.permutation_test(
            (a, b), lambda x, y: abs(welch_t(x, y)), alternative="greater",
            permutation_type="independent", n_resamples=np.inf,
        )
        assert permutation_test(a, b) == pytest.approx(referee.pvalue)


def test_studentized_and_plain_tests_differ_when_noise_is_uneven():
    assert permutation_test(UNEVEN_A, UNEVEN_B) != permutation_test(
        UNEVEN_A, UNEVEN_B, studentized=False
    )


def test_studentized_test_handles_groups_with_no_noise():
    # 3 and 3 runs: the clearest result gives the floor, 2 of 20 shuffles.
    assert permutation_test([1, 1, 1], [2, 2, 2]) == pytest.approx(2 / 20)
    assert permutation_test([1, 1, 1], [1, 1, 1]) == 1


def test_t_interval_matches_scipy_for_independent_runs():
    for a, b in [(A, B), (UNEVEN_A, UNEVEN_B)]:
        referee = scipy_stats.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
        assert t_interval(a, b) == pytest.approx((referee.low, referee.high))


def test_t_interval_matches_scipy_for_paired_runs():
    referee = scipy_stats.ttest_rel(PAIRED_B, PAIRED_A).confidence_interval(0.95)
    assert t_interval(PAIRED_A, PAIRED_B, paired=True) == pytest.approx(
        (referee.low, referee.high)
    )


def test_t_interval_on_the_two_examples():
    assert t_interval(A, B) == pytest.approx((0.14, 1.18), abs=0.005)
    assert t_interval(PAIRED_A, PAIRED_B, paired=True) == pytest.approx((0.40, 0.80), abs=0.005)


def test_t_interval_is_wider_than_the_bootstrap_with_few_runs():
    # The bootstrap interval is too narrow with 5 runs; the t-interval corrects that.
    for a, b, paired in [(A, B, False), (PAIRED_A, PAIRED_B, True)]:
        t_low, t_high = t_interval(a, b, paired)
        boot_low, boot_high = bootstrap_ci(a, b, paired)
        assert t_low < boot_low and boot_high < t_high


def test_t_interval_is_centred_on_the_difference():
    low, high = t_interval(A, B)
    assert (low + high) / 2 == pytest.approx(0.66)


def test_t_interval_widens_with_the_confidence_level():
    low_90, high_90 = t_interval(A, B, confidence=0.90)
    low_99, high_99 = t_interval(A, B, confidence=0.99)
    assert low_99 < low_90 and high_90 < high_99


def test_t_interval_without_any_noise_is_a_single_point():
    assert t_interval([1, 1, 1], [2, 2, 2]) == (1.0, 1.0)
    assert t_interval([1, 2, 3], [2, 3, 4], paired=True) == (1.0, 1.0)


def test_paired_p_value_matches_scipy():
    referee = scipy_stats.permutation_test(
        (PAIRED_A, PAIRED_B), mean_difference, permutation_type="samples", n_resamples=np.inf
    )
    assert permutation_test(PAIRED_A, PAIRED_B, paired=True) == pytest.approx(referee.pvalue)


# --- Sanity checks ---------------------------------------------------------


def test_identical_models_show_no_difference():
    result = compare(A, A)
    assert result.difference == 0
    assert result.p_value == 1
    assert result.cohens_d == 0
    assert result.superiority == 0.5


def test_confidence_interval_contains_the_observed_difference():
    for a, b, paired in [(A, B, False), (PAIRED_A, PAIRED_B, True)]:
        low, high = bootstrap_ci(a, b, paired)
        difference = np.mean(b) - np.mean(a)
        assert low < difference < high


def test_wider_confidence_gives_wider_interval():
    low_90, high_90 = bootstrap_ci(A, B, confidence=0.90)
    low_99, high_99 = bootstrap_ci(A, B, confidence=0.99)
    assert low_99 < low_90 and high_90 < high_99


def test_same_input_gives_same_answer():
    assert bootstrap_ci(A, B) == bootstrap_ci(A, B)


def test_pairing_reveals_the_effect_hidden_by_seed_noise():
    unpaired = permutation_test(PAIRED_A, PAIRED_B, paired=False)
    paired = permutation_test(PAIRED_A, PAIRED_B, paired=True)
    assert unpaired > 0.5  # looks like nothing
    assert paired < 0.1  # consistent win on every seed


def test_many_runs_use_random_shuffles():
    # 15 runs per model has 155 million re-splits: too many to try them all.
    rng = np.random.default_rng(1)
    a = rng.normal(80, 1, size=15)
    b = rng.normal(82, 1, size=15)
    p = permutation_test(a, b)
    assert 0 < p < 0.01
    assert p == permutation_test(a, b)  # still reproducible


def test_compare_bundles_everything():
    result = compare(A, B)
    assert result.difference == pytest.approx(0.66)
    assert result.p_value == pytest.approx(8 / 252)
    assert result.ci_low < 0.66 < result.ci_high
    assert result.summary_b.mean == pytest.approx(84.20)


# --- Bad input gets a clear error ------------------------------------------


def test_too_few_runs_is_rejected():
    with pytest.raises(ValueError, match="at least 2 runs"):
        compare([83.1], B)


def test_paired_needs_equal_lengths():
    with pytest.raises(ValueError, match="same number of runs"):
        compare(A, B[:4], paired=True)


def test_missing_scores_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        compare([83.1, float("nan"), 83.5], B)


def test_infinite_scores_are_rejected():
    with pytest.raises(ValueError, match="not numbers"):
        compare([83.1, float("inf"), 83.5], B)
