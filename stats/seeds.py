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
from scipy import stats as scipy_stats

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
    noise: float  # the run-to-run noise the difference is compared against
    cohens_d: float  # difference divided by noise
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
        if not np.isfinite(scores).all():  # catches missing values and infinity
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


def t_interval(a, b, paired=False, confidence=0.95):
    """Confidence interval for the difference (B minus A). This is the one easyCI shows.

        difference  plus or minus  multiplier x standard error

    The standard error is how much the difference would wobble if the whole
    experiment were repeated. The multiplier comes from the t-distribution:
    it is 1.96 with many runs and grows when runs are few (2.78 for 5 paired
    runs), which makes up for the noise itself being estimated from few runs.

    Unpaired runs use Welch's version, which lets the two models have
    different amounts of noise.
    """
    a, b = _prepare(a, b, paired)
    if paired:
        d = b - a
        difference = d.mean()
        standard_error = d.std(ddof=1) / np.sqrt(len(d))
        freedom = len(d) - 1
    else:
        difference = b.mean() - a.mean()
        part_a = a.var(ddof=1) / len(a)  # A's share of the wobble
        part_b = b.var(ddof=1) / len(b)  # B's share
        standard_error = np.sqrt(part_a + part_b)
        if standard_error > 0:
            # Welch's formula for the "degrees of freedom": roughly, how many
            # runs' worth of information the noise estimate is based on.
            freedom = (part_a + part_b) ** 2 / (
                part_a**2 / (len(a) - 1) + part_b**2 / (len(b) - 1)
            )

    if standard_error == 0:  # no noise at all: the interval is a single point
        return float(difference), float(difference)
    multiplier = scipy_stats.t.ppf(1 - (1 - confidence) / 2, freedom)
    margin = multiplier * standard_error
    return float(difference - margin), float(difference + margin)


def bootstrap_ci(a, b, paired=False, confidence=0.95, n_resamples=10_000, seed=0):
    """An earlier confidence interval, kept for comparison. easyCI shows t_interval.

    Resample the runs many times, compute the difference each time, and keep
    the middle part (95% by default) of all those differences.

    The validation simulation showed that with few runs this interval is too
    narrow: labelled 95%, it contained the truth only 81% to 93% of the time.
    The simulation still measures it next to t_interval.
    """
    a, b = _prepare(a, b, paired)
    rng = np.random.default_rng(seed)  # fixed seed: same input, same answer

    # All resamples are drawn in one step, as a table of positions: each row is
    # one resample, and each entry says which run to pick (repeats allowed).
    # values[picks] then looks up the scores, and .mean(axis=1) averages each row.
    if paired:
        d = b - a  # one difference per seed
        picks = rng.integers(0, len(d), size=(n_resamples, len(d)))
        diffs = d[picks].mean(axis=1)
    else:
        picks_a = rng.integers(0, len(a), size=(n_resamples, len(a)))
        picks_b = rng.integers(0, len(b), size=(n_resamples, len(b)))
        diffs = b[picks_b].mean(axis=1) - a[picks_a].mean(axis=1)

    tail = (1 - confidence) / 2 * 100  # 2.5 when confidence is 0.95
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(low), float(high)


def permutation_test(a, b, paired=False, seed=0, studentized=True):
    """p-value: how often a gap this big appears if the models are identical.

    Unpaired: pool all scores and re-split them into two groups.
    Paired:   flip the sign of each per-seed difference.
    Counts the shuffles whose gap is at least as far from zero as the real one.

    studentized=True (unpaired only) measures each gap as "difference divided
    by its standard error" instead of the raw difference. A shuffle that
    happens to put the noisy scores in the small group then gets a large
    standard error, so its gap counts for less. Without this, the test raises
    too many false alarms when one model is noisier and has fewer runs (17%
    instead of 5% in the validation simulation). studentized=False is the
    earlier version, kept so the simulation can compare the two.

    For paired runs the two versions always give the same p-value, so the
    simpler one is used.
    """
    a, b = _prepare(a, b, paired)
    rng = np.random.default_rng(seed)

    # Each branch below builds a table with one row per shuffle, computes the
    # gap of every row in one step, and counts the rows at least as extreme
    # as the real result.

    if paired:
        d = b - a
        observed = abs(d.mean())
        n_shuffles = 2 ** len(d)
        exact = n_shuffles <= MAX_EXACT
        if exact:
            # Every way to put a plus or a minus in front of each difference.
            signs = np.array(list(product([1, -1], repeat=len(d))))
        else:
            signs = rng.choice([1, -1], size=(N_RANDOM, len(d)))
        gaps = np.abs((signs * d).mean(axis=1))
    else:
        pooled = np.concatenate([a, b])
        # Subtracting the overall mean changes no gap, and keeps the sums of
        # squares below small and accurate.
        pooled = pooled - pooled.mean()
        n, n_b = len(pooled), len(b)
        n_a = n - n_b
        spread = pooled.std()

        def gap(group_b):
            """The gap for each row of group_b (a table: one shuffle per row)."""
            sum_b = group_b.sum(axis=1)
            squares_b = (group_b**2).sum(axis=1)
            mean_b = sum_b / n_b
            mean_a = (pooled.sum() - sum_b) / n_a
            difference = np.abs(mean_b - mean_a)
            if not studentized:
                return difference
            # Variance of each group, from its sum and its sum of squares.
            var_b = (squares_b - n_b * mean_b**2) / (n_b - 1)
            var_a = ((pooled**2).sum() - squares_b - n_a * mean_a**2) / (n_a - 1)
            standard_error = np.sqrt(np.clip(var_a / n_a + var_b / n_b, 0, None))
            # A shuffle whose two groups have no noise at all cannot be
            # divided by its standard error: its gap is "infinitely clear"
            # if the groups differ, and zero if they do not.
            no_noise = standard_error <= 1e-9 * spread
            return np.where(
                no_noise,
                np.where(difference > 1e-9 * spread, np.inf, 0.0),
                difference / np.where(no_noise, 1.0, standard_error),
            )

        observed = gap(pooled[None, n_a:])[0]  # the real B scores are the last n_b
        n_shuffles = comb(n, n_b)
        exact = n_shuffles <= MAX_EXACT
        if exact:
            # Every way to choose which scores are called "B".
            positions = np.array(list(combinations(range(n), n_b)))
            gaps = gap(pooled[positions])
        else:
            # Shuffle the pooled scores N_RANDOM times; the first n_b are "B".
            shuffled = rng.permuted(np.tile(pooled, (N_RANDOM, 1)), axis=1)
            gaps = gap(shuffled[:, :n_b])

    if np.isinf(observed):
        count = int(np.isinf(gaps).sum())
    else:
        count = int((gaps >= observed - TOLERANCE * max(1.0, observed)).sum())
    if exact:
        return count / n_shuffles
    # The +1 counts the real data as one of the shuffles, so p is never 0.
    return (count + 1) / (N_RANDOM + 1)


def run_noise(a, b, paired=False):
    """The typical run-to-run noise that the difference is compared against.

    Paired:   the spread of the per-seed differences.
    Unpaired: the pooled standard deviation of the two models.
    """
    a, b = _prepare(a, b, paired)
    if paired:
        return float((b - a).std(ddof=1))
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    # Pooled standard deviation: the two spreads combined, weighted by size.
    return float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)))


def cohens_d(a, b, paired=False):
    """Effect size: the difference measured in units of run-to-run noise."""
    a, b = _prepare(a, b, paired)
    difference = b.mean() - a.mean()
    noise = run_noise(a, b, paired)

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
    ci_low, ci_high = t_interval(a, b, paired, confidence)
    return Comparison(
        summary_a=summarize(a),
        summary_b=summarize(b),
        difference=float(b.mean() - a.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=permutation_test(a, b, paired),
        noise=run_noise(a, b, paired),
        cohens_d=cohens_d(a, b, paired),
        superiority=probability_of_superiority(a, b, paired, higher_is_better),
        paired=paired,
        higher_is_better=higher_is_better,
    )
