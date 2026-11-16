"""A small, fast version of the validation simulation.

The full simulation (python -m validation.simulate) takes minutes. These
checks take seconds and guard the most important guarantees, so a future
change that breaks them is caught by the normal tests.
"""

from validation.simulate import simulate_examples, simulate_seeds

REPEATS = 400  # few repeats, so the limits below are deliberately loose


def test_false_alarms_stay_near_5_percent_for_independent_runs():
    outcome = simulate_seeds(10, repeats=REPEATS)
    assert outcome["detected"] <= 0.09


def test_false_alarms_stay_near_5_percent_for_paired_runs():
    outcome = simulate_seeds(10, paired=True, repeats=REPEATS)
    assert outcome["detected"] <= 0.09


def test_five_paired_runs_can_never_raise_an_alarm():
    # The p-value floor for 5 paired runs is 0.0625, above the 0.05 bar.
    assert simulate_seeds(5, paired=True, repeats=REPEATS)["detected"] == 0


def test_a_large_true_difference_is_detected():
    outcome = simulate_seeds(10, true_difference=3.0, repeats=REPEATS)
    assert outcome["detected"] >= 0.95


def test_more_runs_give_more_power():
    few = simulate_seeds(5, true_difference=1.0, repeats=REPEATS)
    many = simulate_seeds(20, true_difference=1.0, repeats=REPEATS)
    assert many["detected"] > few["detected"]


def test_coverage_is_reasonable_with_many_runs():
    outcome = simulate_seeds(20, true_difference=1.0, repeats=REPEATS)
    assert 0.88 <= outcome["coverage"] <= 0.99


def test_per_example_false_alarms_stay_near_5_percent():
    outcome = simulate_examples(1000, repeats=REPEATS)
    assert outcome["detected"] <= 0.09
    assert 0.90 <= outcome["coverage"] <= 0.99


def test_per_example_true_difference_is_detected_with_a_large_test_set():
    outcome = simulate_examples(5000, accuracy_b=0.89, repeats=REPEATS)
    assert outcome["detected"] >= 0.95


def test_margins_surround_the_rate():
    outcome = simulate_seeds(10, true_difference=1.0, repeats=REPEATS)
    assert outcome["detected_low"] < outcome["detected"] < outcome["detected_high"]
    assert outcome["repeats"] == REPEATS


def test_the_simulation_is_reproducible():
    assert simulate_seeds(5, repeats=50, seed=3) == simulate_seeds(5, repeats=50, seed=3)
