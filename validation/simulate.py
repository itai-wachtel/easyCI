"""Check easyCI on fake data where the truth is known.

Each simulation repeats the same experiment many times:
    1. generate fake results from models whose true difference we chose,
    2. run easyCI's own functions on them,
    3. record whether easyCI was right.

Three things are counted:
    detected   how often the p-value was below the bar. When the true
               difference is zero this is the false-alarm rate (it should be
               at most 5%); when there is a true difference it is the power.
    coverage   how often the confidence interval contained the true
               difference (it should be 95%).
    real       how often the verdict would be "likely real".

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
from stats.seeds import bootstrap_ci, permutation_test

FOLDER = Path(__file__).parent
CONFIDENCE = 0.95
BAR = 0.05


def _summary(detected, covered, real, repeats):
    """Turn the three counts into rates, each with a margin of error.

    The margin is a Wilson interval: the simulation is itself a sample, so
    its rates are estimates too. Rates are fractions from 0 to 1.
    """
    summary = {"repeats": repeats}
    for name, count in (("detected", detected), ("coverage", covered), ("real", real)):
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

    detected = covered = real = 0
    for repeat in range(repeats):
        if paired:
            seed_effect = rng.normal(0.0, 3.0, runs_a)
            a = seed_effect + noise(runs_a) / np.sqrt(2)
            b = seed_effect + true_difference + noise(runs_a) / np.sqrt(2)
        else:
            a = noise_a * noise(runs_a)
            b = true_difference + noise_b * noise(runs_b)

        p = permutation_test(a, b, paired, seed=repeat)
        low, high = bootstrap_ci(a, b, paired, CONFIDENCE, seed=repeat)

        below_bar = p < BAR
        detected += below_bar
        covered += low <= true_difference <= high
        real += below_bar and (low > 0 or high < 0)
    return _summary(detected, covered, real, repeats)


def simulate_examples(n_examples, accuracy_a=0.85, accuracy_b=0.85, repeats=2000, seed=0):
    """Simulate the per-example mode.

    Both models are right on the same 78% of examples; the rest of each
    model's accuracy comes from examples only that model gets right.
    """
    both = 0.78
    shares = [both, accuracy_a - both, accuracy_b - both, 1 - accuracy_a - accuracy_b + both]
    true_difference = 100 * (accuracy_b - accuracy_a)  # in percentage points
    rng = np.random.default_rng(seed)

    detected = covered = real = 0
    for repeat in range(repeats):
        table = AgreementTable(*(int(count) for count in rng.multinomial(n_examples, shares)))
        p = mcnemar_test(table.only_a, table.only_b)
        low, high = example_bootstrap_ci(table, CONFIDENCE, seed=repeat)

        below_bar = p < BAR
        detected += below_bar
        covered += low <= true_difference <= high
        real += below_bar and (low > 0 or high < 0)
    return _summary(detected, covered, real, repeats)


def run_all(repeats=2000):
    """Run every scenario and return one table with a row per scenario."""
    rows = []

    def add(question, design, runs, true_difference, outcome, **extra):
        rows.append(
            {"question": question, "design": design, "runs": runs,
             "true_difference": true_difference, **extra, **outcome}
        )
        print(f"  {question:<18} {design:<34} runs={runs:<10} "
              f"detected={outcome['detected']:.3f}  coverage={outcome['coverage']:.3f}")

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
        add("awkward data", "independent, skewed noise", runs, 0.0,
            simulate_seeds(runs, skewed=True, repeats=repeats))
        add("awkward data", "independent, B three times noisier", runs, 0.0,
            simulate_seeds(runs, noise_b=3.0, repeats=repeats))
    add("awkward data", "independent, fewer and noisier B runs", "10 and 5", 0.0,
        simulate_seeds(10, 5, noise_b=3.0, repeats=repeats))
    add("awkward data", "independent, more and noisier B runs", "5 and 10", 0.0,
        simulate_seeds(5, 10, noise_b=3.0, repeats=repeats))

    return pd.DataFrame(rows)


def save_charts(results):
    """Save the three main results as images, for the README."""
    import matplotlib

    matplotlib.use("Agg")  # draw to files, without opening a window
    import matplotlib.pyplot as plt

    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#e1e0d9"
    colours = {"independent": blue, "paired": orange}

    def new_chart(title, y_label):
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return fig, ax

    def draw_lines(ax, rows, measure):
        for design, colour in colours.items():
            part = rows[rows.design == design]
            x = range(len(part))
            # Bar lengths below and above each point. clip() removes the tiny
            # negative lengths that rounding produces when a rate is exactly 0.
            below = (100 * (part[measure] - part[f"{measure}_low"])).clip(lower=0)
            above = (100 * (part[f"{measure}_high"] - part[measure])).clip(lower=0)
            ax.errorbar(
                x, 100 * part[measure],
                yerr=[below, above],
                color=colour, marker="o", markersize=6, linewidth=2, capsize=3,
                label=f"{design} runs",
            )
            for position, runs in zip(x, part.runs):
                ax.annotate(f"{runs} runs", (position, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -14 if design == "independent" else -27),
                            textcoords="offset points", ha="center", fontsize=8, color=colour)
        ax.set_xticks([])

    def finish(fig, ax, name):
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(FOLDER / name)
        plt.close(fig)

    fig, ax = new_chart("False alarms when there is no true difference", "False-alarm rate (%)")
    draw_lines(ax, results[results.question == "false alarms"], "detected")
    ax.axhline(5, color=ink, linestyle="--", linewidth=1)
    ax.annotate("5% limit", (0, 5), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_ylim(0, 8)
    finish(fig, ax, "false_alarms.png")

    fig, ax = new_chart("How often the 95% interval contains the truth", "Coverage (%)")
    draw_lines(ax, results[results.question == "true difference"], "coverage")
    ax.axhline(95, color=ink, linestyle="--", linewidth=1)
    ax.annotate("95% target", (0, 95), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_ylim(60, 100)
    finish(fig, ax, "coverage.png")

    fig, ax = new_chart("Power at the number of runs the calculator recommends", "Power (%)")
    draw_lines(ax, results[results.question == "calculator"], "detected")
    ax.axhline(80, color=ink, linestyle="--", linewidth=1)
    ax.annotate("80% target", (0, 80), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_ylim(40, 100)
    finish(fig, ax, "calculator.png")


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
    print(f"\nSaved results.csv and three charts in {FOLDER}")
