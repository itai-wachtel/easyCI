"""Compare two models when each one has several scores (one per training run).

Every function takes:
    a, b    - the scores of model A and model B (lists of numbers)
    paired  - True if run i of A and run i of B shared the same seed,
              False if the runs are independent

All differences are reported as "B minus A".
"""

from dataclasses import dataclass
from itertools import combinations, product
from math import comb

import numpy as np

# The permutation test tries every possible shuffle when there are at most
# this many. Above it, it tries N_RANDOM random shuffles instead.
MAX_EXACT = 100_000
N_RANDOM = 10_000

# Computers store decimals with tiny rounding errors (0.1 + 0.2 is not exactly
# 0.3). This tolerance stops such errors from changing a "greater or equal".
TOLERANCE = 1e-9


@dataclass
class Summary:
    """The basic description of one model's runs."""

    mean: float
    std: float
    n: int


@dataclass
class Comparison:
    """Everything easyCI knows about A versus B."""

    summary_a: Summary
    summary_b: Summary
    difference: float  # mean of B minus mean of A
    ci_low: float  # confidence interval for the difference
    ci_high: float
    confidence: float  # e.g. 0.95
    p_value: float
    cohens_d: float
    superiority: float  # how often B is the better model, from 0 to 1
    paired: bool
    higher_is_better: bool


def _prepare(a, b, paired):
    """Turn the inputs into numpy arrays and reject input we cannot analyse."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    for name, scores in (("A", a), ("B", b)):
        if scores.ndim != 1:
            raise ValueError(f"Model {name}: scores must be a simple list of numbers.")
        if len(scores) < 2:
            raise ValueError(f"Model {name}: at least 2 runs are needed.")
        if np.isnan(scores).any():
            raise ValueError(f"Model {name}: some scores are missing or not numbers.")
    if paired and len(a) != len(b):
        raise ValueError(
            "Paired comparison needs the same number of runs for both models "
            f"(got {len(a)} and {len(b)})."
        )
    return a, b


def summarize(scores):
    """Mean, standard deviation and number of runs for one model."""
    scores = np.asarray(scores, dtype=float)
    # ddof=1 divides by (n - 1): the standard formula for a sample.
    return Summary(mean=float(scores.mean()), std=float(scores.std(ddof=1)), n=len(scores))


def bootstrap_ci(a, b, paired=False, confidence=0.95, n_resamples=10_000, seed=0):
    """Confidence interval for the difference (B minus A).

    Resample the runs many times, compute the difference each time, and keep
    the middle part (95% by default) of all those differences.
    """
    a, b = _prepare(a, b, paired)
    rng = np.random.default_rng(seed)  # fixed seed: same input, same answer
    diffs = np.empty(n_resamples)

    if paired:
        d = b - a  # one difference per seed
        for i in range(n_resamples):
            diffs[i] = rng.choice(d, size=len(d), replace=True).mean()
    else:
        for i in range(n_resamples):
            new_a = rng.choice(a, size=len(a), replace=True)
            new_b = rng.choice(b, size=len(b), replace=True)
            diffs[i] = new_b.mean() - new_a.mean()

    tail = (1 - confidence) / 2 * 100  # 2.5 when confidence is 0.95
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(low), float(high)


def permutation_test(a, b, paired=False, seed=0):
    """p-value: how often a gap this big appears if the models are identical.

    Unpaired: pool all scores and re-split them into two groups.
    Paired:   flip the sign of each per-seed difference.
    Counts the shuffles whose gap is at least as far from zero as the real one.
    """
    a, b = _prepare(a, b, paired)
    rng = np.random.default_rng(seed)

    if paired:
        d = b - a
        observed = abs(d.mean())
        n_shuffles = 2 ** len(d)
        if n_shuffles <= MAX_EXACT:
            count = 0
            for signs in product([1, -1], repeat=len(d)):
                if abs((d * signs).mean()) >= observed - TOLERANCE:
                    count += 1
            return count / n_shuffles
        count = 0
        for _ in range(N_RANDOM):
            signs = rng.choice([1, -1], size=len(d))
            if abs((d * signs).mean()) >= observed - TOLERANCE:
                count += 1
        return (count + 1) / (N_RANDOM + 1)

    pooled = np.concatenate([a, b])
    total = pooled.sum()
    n, n_b = len(pooled), len(b)
    n_a = n - n_b
    observed = abs(b.mean() - a.mean())

    def gap(sum_b):
        """Difference of means when the 'B' group adds up to sum_b."""
        return abs(sum_b / n_b - (total - sum_b) / n_a)

    n_shuffles = comb(n, n_b)
    if n_shuffles <= MAX_EXACT:
        count = 0
        for positions in combinations(range(n), n_b):
            if gap(pooled[list(positions)].sum()) >= observed - TOLERANCE:
                count += 1
        return count / n_shuffles
    count = 0
    for _ in range(N_RANDOM):
        shuffled = rng.permutation(pooled)
        if gap(shuffled[:n_b].sum()) >= observed - TOLERANCE:
            count += 1
    # The +1 counts the real data as one of the shuffles, so p is never 0.
    return (count + 1) / (N_RANDOM + 1)


def cohens_d(a, b, paired=False):
    """Effect size: the difference measured in units of run-to-run noise."""
    a, b = _prepare(a, b, paired)
    difference = b.mean() - a.mean()

    if paired:
        noise = (b - a).std(ddof=1)  # spread of the per-seed differences
    else:
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        n_a, n_b = len(a), len(b)
        # Pooled standard deviation: the two spreads combined, weighted by size.
        noise = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if noise == 0:  # no noise at all: the gap is either nothing or "infinite"
        return 0.0 if difference == 0 else float(np.sign(difference) * np.inf)
    return float(difference / noise)


def probability_of_superiority(a, b, paired=False, higher_is_better=True):
    """How often B beats A, from 0 to 1. A tie counts as half a win.

    Unpaired: over every match-up of one A run against one B run.
    Paired:   over the seeds.
    """
    a, b = _prepare(a, b, paired)
    if not higher_is_better:  # for loss or error, the lower score wins
        a, b = -a, -b

    if paired:
        wins, ties, total = (b > a).sum(), (b == a).sum(), len(a)
    else:
        # b[:, None] - a builds a table with every B run minus every A run.
        table = b[:, None] - a
        wins, ties, total = (table > 0).sum(), (table == 0).sum(), table.size
    return float((wins + 0.5 * ties) / total)


def compare(a, b, paired=False, higher_is_better=True, confidence=0.95):
    """Run every analysis and return the results in one bundle."""
    a, b = _prepare(a, b, paired)
    ci_low, ci_high = bootstrap_ci(a, b, paired, confidence)
    return Comparison(
        summary_a=summarize(a),
        summary_b=summarize(b),
        difference=float(b.mean() - a.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=permutation_test(a, b, paired),
        cohens_d=cohens_d(a, b, paired),
        superiority=probability_of_superiority(a, b, paired, higher_is_better),
        paired=paired,
        higher_is_better=higher_is_better,
    )
