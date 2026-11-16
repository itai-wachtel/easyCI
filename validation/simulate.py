"""Check easyCI on fake data where the truth is known.

Each simulation repeats the same experiment many times:
    1. generate fake results from models whose true difference we chose,
    2. run easyCI's own functions on them,
    3. record whether easyCI was right.

These things are counted:
    detected   how often the p-value was below the bar. When the true
               difference is zero this is the false-alarm rate (it should be
               at most 5%); when there is a true difference it is the power.
    coverage   how often the confidence interval contained the true
               difference (it should be 95%).
    real       how often the verdict would be "likely real".

The per-seed simulation also measures the two methods easyCI used at first,
on exactly the same fake data, so the old and new methods can be compared:
    detected_plain       the permutation test on the raw difference
    coverage_bootstrap   the percentile bootstrap interval

Run every scenario and save the results and charts with:
    python -m validation.simulate
Redraw the charts from the saved results, without simulating again:
    python -m validation.simulate --charts-only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats.examples import AgreementTable, mcnemar_test, wilson_interval
from stats.examples import bootstrap_ci as example_bootstrap_ci
from stats.planning import runs_needed
from stats.seeds import bootstrap_ci, permutation_test, t_interval

FOLDER = Path(__file__).parent
CONFIDENCE = 0.95
BAR = 0.05


def _summary(counts, repeats):
    """Turn counts into rates, each with a margin of error.

    The margin is a Wilson interval: the simulation is itself a sample, so
    its rates are estimates too. Rates are fractions from 0 to 1.
    """
    summary = {"repeats": repeats}
    for name, count in counts.items():
        low, high = wilson_interval(count, repeats)
        summary[name] = count / repeats
        summary[f"{name}_low"] = low / 100
        summary[f"{name}_high"] = high / 100
    return summary


def simulate_seeds(
    runs_a,
    runs_b=None,
    true_difference=0.0,
    paired=False,
    noise_a=1.0,
    noise_b=1.0,
    skewed=False,
    repeats=2000,
    seed=0,
):
    """Simulate the per-seed mode.

    Independent runs: A's scores are drawn around 0 and B's around
    `true_difference`, with standard deviations noise_a and noise_b.

    Paired runs: every seed adds its own large effect (standard deviation 3)
    to both models, and the per-seed differences have standard deviation 1.

    skewed=True draws lopsided noise (many small values, a few large ones)
    instead of the bell curve.
    """
    runs_b = runs_a if runs_b is None else runs_b
    rng = np.random.default_rng(seed)

    def noise(size):
        if skewed:
            return rng.exponential(1.0, size) - 1.0  # mean 0, standard deviation 1
        return rng.normal(0.0, 1.0, size)

    counts = dict(detected=0, coverage=0, real=0, detected_plain=0, coverage_bootstrap=0)
    for repeat in range(repeats):
        if paired:
            seed_effect = rng.normal(0.0, 3.0, runs_a)
            a = seed_effect + noise(runs_a) / np.sqrt(2)
            b = seed_effect + true_difference + noise(runs_a) / np.sqrt(2)
        else:
            a = noise_a * noise(runs_a)
            b = true_difference + noise_b * noise(runs_b)

        # The methods easyCI uses now.
        p = permutation_test(a, b, paired, seed=repeat)
        low, high = t_interval(a, b, paired, CONFIDENCE)
        below_bar = p < BAR
        counts["detected"] += below_bar
        counts["coverage"] += low <= true_difference <= high
        counts["real"] += below_bar and (low > 0 or high < 0)

        # The methods it used at first. (For paired runs the two tests are
        # always identical, so the test is not run twice.)
        p_plain = p if paired else permutation_test(a, b, seed=repeat, studentized=False)
        old_low, old_high = bootstrap_ci(a, b, paired, CONFIDENCE, seed=repeat)
        counts["detected_plain"] += p_plain < BAR
        counts["coverage_bootstrap"] += old_low <= true_difference <= old_high
    return _summary(counts, repeats)


def simulate_examples(n_examples, accuracy_a=0.85, accuracy_b=0.85, repeats=2000, seed=0):
    """Simulate the per-example mode.

    Both models are right on the same 78% of examples; the rest of each
    model's accuracy comes from examples only that model gets right.
    """
    both = 0.78
    shares = [both, accuracy_a - both, accuracy_b - both, 1 - accuracy_a - accuracy_b + both]
    true_difference = 100 * (accuracy_b - accuracy_a)  # in percentage points
    rng = np.random.default_rng(seed)

    counts = dict(detected=0, coverage=0, real=0)
    for repeat in range(repeats):
        table = AgreementTable(*(int(count) for count in rng.multinomial(n_examples, shares)))
        p = mcnemar_test(table.only_a, table.only_b)
        low, high = example_bootstrap_ci(table, CONFIDENCE, seed=repeat)

        below_bar = p < BAR
        counts["detected"] += below_bar
        counts["coverage"] += low <= true_difference <= high
        counts["real"] += below_bar and (low > 0 or high < 0)
    return _summary(counts, repeats)


def run_all(repeats=2000):
    """Run every scenario and return one table with a row per scenario."""
    rows = []

    def add(question, design, runs, true_difference, outcome):
        rows.append(
            {"question": question, "design": design, "runs": runs,
             "true_difference": true_difference, **outcome}
        )
        line = (f"  {design:<38} runs={runs!s:<9} detected={outcome['detected']:.3f} "
                f"coverage={outcome['coverage']:.3f}")
        if "detected_plain" in outcome:
            line += (f"   earlier methods: detected={outcome['detected_plain']:.3f} "
                     f"coverage={outcome['coverage_bootstrap']:.3f}")
        print(line, flush=True)

    print("1. No true difference: how often does easyCI raise a false alarm?")
    for runs in (3, 5, 10, 20):
        add("false alarms", "independent", runs, 0.0,
            simulate_seeds(runs, repeats=repeats))
    for runs in (5, 6, 10, 20):
        add("false alarms", "paired", runs, 0.0,
            simulate_seeds(runs, paired=True, repeats=repeats))

    print("2. A true difference of 1 (as big as the noise): coverage and power")
    for runs in (3, 5, 10, 20):
        add("true difference", "independent", runs, 1.0,
            simulate_seeds(runs, true_difference=1.0, repeats=repeats))
    for runs in (5, 6, 10, 20):
        add("true difference", "paired", runs, 1.0,
            simulate_seeds(runs, true_difference=1.0, paired=True, repeats=repeats))

    print("3. The calculator: is the power really 80% at the runs it recommends?")
    for paired in (False, True):
        for difference in (0.5, 0.8, 1.0, 1.5):
            runs = runs_needed(difference, 1.0, paired, CONFIDENCE)
            add("calculator", "paired" if paired else "independent", runs, difference,
                simulate_seeds(runs, true_difference=difference, paired=paired, repeats=repeats))

    print("4. Per-example mode")
    for n_examples in (200, 1000, 5000):
        add("false alarms", "per-example", n_examples, 0.0,
            simulate_examples(n_examples, repeats=repeats))
    for n_examples in (200, 1000, 5000):
        add("true difference", "per-example", n_examples, 4.0,
            simulate_examples(n_examples, accuracy_b=0.89, repeats=repeats))

    print("5. Awkward data, no true difference")
    for runs in (5, 10):
        add("awkward data", "skewed noise", runs, 0.0,
            simulate_seeds(runs, skewed=True, repeats=repeats))
        add("awkward data", "B three times noisier", runs, 0.0,
            simulate_seeds(runs, noise_b=3.0, repeats=repeats))
    add("awkward data", "B noisier, with fewer runs", "10 and 5", 0.0,
        simulate_seeds(10, 5, noise_b=3.0, repeats=repeats))
    add("awkward data", "B noisier, with more runs", "5 and 10", 0.0,
        simulate_seeds(5, 10, noise_b=3.0, repeats=repeats))

    return pd.DataFrame(rows)


def save_charts(results):
    """Save the main results as images, for the README."""
    import matplotlib

    matplotlib.use("Agg")  # draw to files, without opening a window
    import matplotlib.pyplot as plt

    blue, orange, grey, ink, grid = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#e1e0d9"

    def style(ax, title, y_label=None):
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def line(ax, rows, measure, colour, label):
        """One line with a point per scenario and a bar for the margin of error."""
        # Bar lengths below and above each point. clip() removes the tiny
        # negative lengths that rounding produces when a rate is exactly 0.
        below = (100 * (rows[measure] - rows[f"{measure}_low"])).clip(lower=0)
        above = (100 * (rows[f"{measure}_high"] - rows[measure])).clip(lower=0)
        ax.errorbar(
            range(len(rows)), 100 * rows[measure], yerr=[below, above],
            color=colour, marker="o", markersize=6, linewidth=2, capsize=3, label=label,
        )
        ax.set_xticks(range(len(rows)), [f"{runs} runs" for runs in rows.runs], fontsize=9)

    def target(ax, value, label):
        ax.axhline(value, color=ink, linestyle="--", linewidth=1)
        ax.annotate(label, (0, value), xytext=(4, 4), textcoords="offset points", fontsize=8)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FOLDER / name)
        plt.close(fig)

    # 1. Coverage: the earlier interval against the one used now.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150, sharey=True)
    for ax, design in zip(axes, ("independent", "paired")):
        rows = results[(results.question == "true difference") & (results.design == design)]
        line(ax, rows, "coverage_bootstrap", grey, "earlier: bootstrap interval")
        line(ax, rows, "coverage", blue, "now: t-interval")
        target(ax, 95, "95% target")
        style(ax, f"{design.capitalize()} runs", "Coverage (%)" if design == "independent" else None)
        ax.set_ylim(75, 100)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("How often the 95% confidence interval contains the truth",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "coverage.png")

    # 2. False alarms with ordinary data.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150, sharey=True)
    for ax, design, colour in zip(axes, ("independent", "paired"), (blue, orange)):
        rows = results[(results.question == "false alarms") & (results.design == design)]
        line(ax, rows, "detected", colour, None)
        target(ax, 5, "5% limit")
        style(ax, f"{design.capitalize()} runs",
              "False-alarm rate (%)" if design == "independent" else None)
        ax.set_ylim(0, 8)
    fig.suptitle("False alarms when there is no true difference", x=0.01, ha="left", fontsize=12)
    save(fig, "false_alarms.png")

    # 3. False alarms with awkward data: the earlier test against the one used now.
    rows = results[results.question == "awkward data"].iloc[::-1]
    names = [f"{design}\n({runs} runs)" for design, runs in zip(rows.design, rows.runs)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    places = np.arange(len(rows))
    ax.barh(places + 0.19, 100 * rows.detected_plain, height=0.34, color=grey,
            label="earlier: raw difference")
    ax.barh(places - 0.19, 100 * rows.detected, height=0.34, color=blue,
            label="now: studentized")
    for place, plain, now in zip(places, rows.detected_plain, rows.detected):
        ax.annotate(f"{100 * plain:.1f}%", (100 * plain, place + 0.19), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8)
        ax.annotate(f"{100 * now:.1f}%", (100 * now, place - 0.19), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.axvline(5, color=ink, linestyle="--", linewidth=1)
    ax.annotate("5% limit", (5, len(rows) - 0.5), xytext=(3, 0),
                textcoords="offset points", fontsize=8)
    ax.set_yticks(places, names, fontsize=8)
    ax.set_xlabel("False-alarm rate (%)")
    ax.set_title("False alarms with awkward data, no true difference",
                 loc="left", fontsize=11, color=ink)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "awkward_data.png")

    # 4. The calculator.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150, sharey=True)
    for ax, design, colour in zip(axes, ("independent", "paired"), (blue, orange)):
        rows = results[(results.question == "calculator") & (results.design == design)]
        line(ax, rows, "detected", colour, None)
        target(ax, 80, "80% target")
        style(ax, f"{design.capitalize()} runs", "Power (%)" if design == "independent" else None)
        ax.set_ylim(40, 100)
    fig.suptitle("Power at the number of runs the calculator recommends",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "calculator.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=2000, help="experiments per scenario")
    parser.add_argument(
        "--charts-only", action="store_true", help="redraw the charts from the saved results.csv"
    )
    arguments = parser.parse_args()

    if arguments.charts_only:
        table = pd.read_csv(FOLDER / "results.csv")
    else:
        table = run_all(arguments.repeats)
        table.to_csv(FOLDER / "results.csv", index=False)
    save_charts(table)
    print(f"\nSaved results.csv and four charts in {FOLDER}")
