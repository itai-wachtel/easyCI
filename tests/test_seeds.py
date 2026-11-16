"""Checks that the maths in stats/seeds.py gives the right answers."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from stats.seeds import (
    bootstrap_ci,
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


def test_unpaired_p_value_matches_scipy():
    referee = scipy_stats.permutation_test(
        (A, B), mean_difference, permutation_type="independent", n_resamples=np.inf
    )
    assert permutation_test(A, B) == pytest.approx(referee.pvalue)


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
