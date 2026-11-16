"""Planning an experiment: how many training runs are enough?"""

from math import ceil, comb
from statistics import NormalDist

from stats.seeds import MAX_EXACT, N_RANDOM


def smallest_possible_p(n_a, n_b, paired=False):
    """The lowest p-value the permutation test can ever give for this many runs.

    The p-value is "extreme shuffles / all shuffles". With few runs there are
    few shuffles, so even the most clear-cut result has a p-value above zero.
    """
    shuffles = 2**n_a if paired else comb(n_a + n_b, n_a)
    if shuffles > MAX_EXACT:  # the test uses random shuffles instead
        return 1 / (N_RANDOM + 1)
    # The most extreme result usually has a mirror image (A and B swapped)
    # that is just as extreme, so two shuffles count. With groups of
    # different sizes there is no mirror image.
    extreme = 2 if (paired or n_a == n_b) else 1
    return extreme / shuffles


def fewest_runs_that_can_pass(paired=False, confidence=0.95):
    """The fewest runs per model for which a p-value below the bar is possible."""
    bar = 1 - confidence
    runs = 2
    while smallest_possible_p(runs, runs, paired) >= bar:
        runs += 1
    return runs


def runs_needed(difference, noise, paired=False, confidence=0.95, power=0.80):
    """Runs per model needed to detect a true difference of this size.

    "Detect" means: if the true difference really is this big, the test passes
    the bar with probability `power` (80% by default).

    The noise of an average shrinks with the square root of the number of
    runs, which leads to the standard formula used here. At 95% confidence
    and 80% power it is about 16 / d**2 for independent runs, where d is the
    difference divided by the noise.

    Returns None if the difference is zero (no number of runs can detect it).
    """
    if difference == 0:
        return None
    least = fewest_runs_that_can_pass(paired, confidence)
    if noise == 0:  # no noise at all: any difference shows immediately
        return least

    d = abs(difference) / noise
    # How many standard errors the difference must clear: one part for the
    # bar (1.96 at 95% confidence) and one for the power (0.84 at 80%).
    z_bar = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z_power = NormalDist().inv_cdf(power)
    # Independent runs: both averages are noisy, hence the factor 2.
    # Paired runs: there is one list of differences, so the factor is 1.
    factor = 1 if paired else 2
    runs = ceil(factor * (z_bar + z_power) ** 2 / d**2)
    return max(runs, least)
