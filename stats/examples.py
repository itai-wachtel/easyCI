"""Compare two models that were each trained once, example by example.

The input is, for every test example, whether each model answered correctly.
Both models must have been tested on the same examples, in the same order.

All accuracies and differences are in percent (0 to 100). Differences are
"B minus A", in percentage points.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass
class AgreementTable:
    """The test examples sorted into four groups."""

    both_correct: int
    only_a: int  # A correct, B wrong
    only_b: int  # B correct, A wrong
    both_wrong: int

    @property
    def n(self):
        return self.both_correct + self.only_a + self.only_b + self.both_wrong


@dataclass
class ExampleComparison:
    """Everything easyCI knows about A versus B in per-example mode."""

    table: AgreementTable
    accuracy_a: float
    accuracy_b: float
    interval_a: tuple  # Wilson interval of A's accuracy (low, high)
    interval_b: tuple
    difference: float  # accuracy of B minus accuracy of A
    ci_low: float  # confidence interval for the difference
    ci_high: float
    confidence: float
    p_value: float  # McNemar's test


def agreement_table(correct_a, correct_b):
    """Count how many examples fall into each of the four groups."""
    a = np.asarray(correct_a)
    b = np.asarray(correct_b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Each model needs one simple list of right/wrong answers.")
    if len(a) != len(b):
        raise ValueError(
            "Both models need an answer for every test example "
            f"(got {len(a)} and {len(b)})."
        )
    if len(a) == 0:
        raise ValueError("There are no test examples.")
    for name, answers in (("A", a), ("B", b)):
        if not np.isin(answers, [0, 1]).all():
            raise ValueError(f"Model {name}: answers must be 1 (correct) or 0 (wrong).")
    a = a.astype(bool)
    b = b.astype(bool)
    return AgreementTable(
        both_correct=int((a & b).sum()),
        only_a=int((a & ~b).sum()),
        only_b=int((~a & b).sum()),
        both_wrong=int((~a & ~b).sum()),
    )


def wilson_interval(correct, n, confidence=0.95):
    """Confidence interval for one accuracy, in percent.

    Like "estimate plus or minus a margin", but adjusted so that it stays
    sensible for small test sets and never leaves the range 0 to 100.
    """
    # z is how many standard errors wide the interval is: 1.96 for 95%.
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = correct / n
    shrink = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / shrink
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / shrink
    return float(100 * (centre - margin)), float(100 * (centre + margin))


def mcnemar_test(only_a, only_b):
    """p-value: is the split of the disagreements more lopsided than coin flips?

    If the models were equally good, each example they disagree on would be
    equally likely to go to A or to B, like a fair coin. The p-value is the
    chance that a fair coin gives a split at least as uneven as the real one.
    """
    flips = only_a + only_b
    if flips == 0:  # the models never disagree
        return 1.0
    smaller = min(only_a, only_b)

    # Count the ways to get 0, 1, ..., `smaller` heads in `flips` coin flips.
    ways = 0
    term = 1  # ways to get exactly 0 heads
    for heads in range(smaller + 1):
        ways += term
        term = term * (flips - heads) // (heads + 1)  # ways to get one more head

    # Each exact sequence of flips has chance 1 / 2**flips. Doubling covers
    # the same unevenness in the other direction.
    return min(1.0, 2 * ways / 2**flips)


def bootstrap_ci(table, confidence=0.95, n_resamples=10_000, seed=0):
    """Confidence interval for the accuracy difference (B minus A), in points.

    Drawing n examples with replacement only changes how many land in each of
    the four groups, so we draw those four counts directly. Each example keeps
    both models' answers, which preserves the pairing.
    """
    rng = np.random.default_rng(seed)
    n = table.n
    shares = np.array([table.both_correct, table.only_a, table.only_b, table.both_wrong]) / n
    counts = rng.multinomial(n, shares, size=n_resamples)  # one row per resample
    diffs = 100 * (counts[:, 2] - counts[:, 1]) / n  # (only B - only A) / n

    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(low), float(high)


def compare_examples(correct_a, correct_b, confidence=0.95):
    """Run every analysis and return the results in one bundle."""
    table = agreement_table(correct_a, correct_b)
    n = table.n
    right_a = table.both_correct + table.only_a
    right_b = table.both_correct + table.only_b
    ci_low, ci_high = bootstrap_ci(table, confidence)
    return ExampleComparison(
        table=table,
        accuracy_a=100 * right_a / n,
        accuracy_b=100 * right_b / n,
        interval_a=wilson_interval(right_a, n, confidence),
        interval_b=wilson_interval(right_b, n, confidence),
        difference=100 * (right_b - right_a) / n,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=mcnemar_test(table.only_a, table.only_b),
    )
