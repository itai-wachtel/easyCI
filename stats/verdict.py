"""Turn the numbers into words: a verdict, cautions, and a sentence for a paper.

Nothing here does new statistics. These functions only read the results that
stats/seeds.py and stats/examples.py produced.
"""

from dataclasses import dataclass

from stats.planning import fewest_runs_that_can_pass, smallest_possible_p

MULTIPLE_COMPARISONS_NOTE = (
    "If you compared many models or settings and are reporting only the best "
    "pair, this p-value is too optimistic."
)


@dataclass
class Verdict:
    kind: str  # "real", "borderline" or "none"
    title: str  # the headline, for example "Likely real: Model B is better"
    reason: str  # one or two sentences saying why this verdict was chosen


def format_p(p):
    """Show a p-value with three meaningful digits: 0.0317, 0.128."""
    return "< 0.001" if p < 0.001 else f"{p:.3g}"


def judge(result, name_a, name_b, higher_is_better=True, more_of="runs"):
    """Decide the verdict from the p-value and the confidence interval.

    Likely real:  the p-value is below the bar AND the interval excludes zero.
    Borderline:   only one of the two holds, or the p-value is within twice
                  the bar (0.05 to 0.10 at 95% confidence).
    No evidence:  everything else.

    `more_of` is what the user could collect more of: "runs" or "test examples".
    """
    bar = round(1 - result.confidence, 10)  # 0.05 at 95% confidence
    percent = round(result.confidence * 100)
    p = format_p(result.p_value)
    below_bar = result.p_value < bar
    excludes_zero = result.ci_low > 0 or result.ci_high < 0

    b_is_better = (result.difference > 0) == higher_is_better
    winner, loser = (name_b, name_a) if b_is_better else (name_a, name_b)

    if below_bar and excludes_zero:
        return Verdict(
            kind="real",
            title=f"Likely real: {winner} is better than {loser}",
            reason=(
                f"The p-value ({p}) is below {bar:g} and the {percent}% confidence "
                "interval does not include zero, so luck is an unlikely explanation."
            ),
        )

    if below_bar or excludes_zero or result.p_value < 2 * bar:
        if below_bar:
            why = (
                f"The p-value ({p}) is below {bar:g}, but the {percent}% confidence "
                "interval includes zero: the two checks disagree."
            )
        elif excludes_zero:
            why = (
                f"The {percent}% confidence interval does not include zero, but the "
                f"p-value ({p}) is not below {bar:g}: the two checks disagree."
            )
        else:
            why = f"The p-value ({p}) is close to {bar:g}, but not below it."
        return Verdict(
            kind="borderline",
            title=f"Borderline: {winner} may be better, but this is not proven",
            reason=f"{why} More {more_of} would settle it.",
        )

    return Verdict(
        kind="none",
        title="No evidence of a difference",
        reason=(
            f"The p-value ({p}) is well above {bar:g} and the {percent}% confidence "
            "interval includes zero. This does not prove the models are equally "
            "good: it means this data cannot tell them apart."
        ),
    )


def seed_cautions(result, name_a, name_b):
    """Weak spots in per-seed data, as a list of plain sentences."""
    cautions = []
    n_a, n_b = result.summary_a.n, result.summary_b.n
    bar = round(1 - result.confidence, 10)

    floor = smallest_possible_p(n_a, n_b, result.paired)
    if floor >= bar:
        runs = f"{n_a} paired runs" if result.paired else f"{n_a} and {n_b} runs"
        enough = fewest_runs_that_can_pass(result.paired, result.confidence)
        cautions.append(
            f"With {runs}, the p-value can never go below {format_p(floor)}, so no "
            f"result could pass the {bar:g} bar. Use at least {enough} runs per model."
        )
    if min(n_a, n_b) < 5:
        cautions.append(
            "With fewer than 5 runs per model, the confidence interval is rough "
            "and tends to be too narrow."
        )
    for name, summary in ((name_a, result.summary_a), (name_b, result.summary_b)):
        if summary.std == 0:
            cautions.append(
                f"All runs of {name} have exactly the same score. Check that the "
                "random seed really changed between runs."
            )
    return cautions


def example_cautions(result):
    """Weak spots in per-example data, as a list of plain sentences."""
    cautions = []
    disagreements = result.table.only_a + result.table.only_b
    if result.table.n < 100:
        cautions.append(
            f"The test set has only {result.table.n} examples, so both accuracies "
            "are very uncertain."
        )
    if disagreements < 25:
        cautions.append(
            f"The models disagree on only {disagreements} examples. McNemar's test "
            "has little to work with, so a real difference could easily be missed."
        )
    return cautions


def _winner_first(result, name_a, name_b, higher_is_better):
    """Return (winner, loser, low, high) with the interval as "winner minus loser"."""
    b_is_better = (result.difference > 0) == higher_is_better
    winner, loser = (name_b, name_a) if b_is_better else (name_a, name_b)
    low, high = result.ci_low, result.ci_high
    if result.difference < 0:  # report the size of the gap as a positive number
        low, high = -high, -low
    return winner, loser, low, high


def seed_sentence(result, verdict, name_a, name_b, decimals=2):
    """One sentence reporting a per-seed comparison, ready for a paper."""
    percent = round(result.confidence * 100)
    n_a, n_b = result.summary_a.n, result.summary_b.n
    if result.paired:
        test, runs = "two-sided paired permutation test", f"n = {n_a} paired runs"
    elif n_a == n_b:
        test, runs = "two-sided permutation test", f"n = {n_a} runs per model"
    else:
        test, runs = "two-sided permutation test", f"n = {n_a} and {n_b} runs"
    p = format_p(result.p_value)
    p = f"p {p}" if p.startswith("<") else f"p = {p}"

    if verdict.kind == "real":
        winner, loser, low, high = _winner_first(
            result, name_a, name_b, result.higher_is_better
        )
        details = (
            f"({percent}% bootstrap CI [{low:.{decimals}f}, {high:.{decimals}f}]; "
            f"{test}, {p}; {runs})"
        )
        return (
            f"{winner} outperforms {loser} by {abs(result.difference):.{decimals}f} "
            f"on average {details}."
        )

    details = (
        f"({percent}% bootstrap CI [{result.ci_low:.{decimals}f}, "
        f"{result.ci_high:.{decimals}f}]; {test}, {p}; {runs})"
    )
    status = "borderline" if verdict.kind == "borderline" else "not statistically significant"
    return (
        f"The difference between {name_b} and {name_a} "
        f"({result.difference:+.{decimals}f} on average) is {status} {details}."
    )


def example_sentence(result, verdict, name_a, name_b):
    """One sentence reporting a per-example comparison, ready for a paper."""
    percent = round(result.confidence * 100)
    p = format_p(result.p_value)
    p = f"p {p}" if p.startswith("<") else f"p = {p}"
    examples = f"n = {result.table.n:,} test examples"

    if verdict.kind == "real":
        winner, loser, low, high = _winner_first(result, name_a, name_b, True)
        return (
            f"{winner} outperforms {loser} by {abs(result.difference):.2f} percentage "
            f"points ({percent}% bootstrap CI [{low:.2f}, {high:.2f}]; "
            f"McNemar's exact test, {p}; {examples})."
        )

    status = "borderline" if verdict.kind == "borderline" else "not statistically significant"
    return (
        f"The difference between {name_b} and {name_a} "
        f"({result.difference:+.2f} percentage points) is {status} "
        f"({percent}% bootstrap CI [{result.ci_low:.2f}, {result.ci_high:.2f}]; "
        f"McNemar's exact test, {p}; {examples})."
    )
