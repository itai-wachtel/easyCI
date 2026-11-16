"""Checks of the verdict, the cautions and the paper sentence."""

from types import SimpleNamespace

from stats.examples import compare_examples
from stats.seeds import compare
from stats.verdict import (
    example_cautions,
    example_sentence,
    format_p,
    judge,
    seed_cautions,
    seed_sentence,
)

# The three examples used throughout the project.
A = [83.1, 83.9, 83.5, 84.0, 83.2]
B = [84.2, 83.8, 84.5, 84.1, 84.4]
PAIRED_A = [80.0, 85.0, 82.0, 88.0, 84.0]
PAIRED_B = [80.6, 85.5, 82.7, 88.4, 84.8]


def answers(both_correct, only_a, only_b, both_wrong):
    a = [1] * both_correct + [1] * only_a + [0] * only_b + [0] * both_wrong
    b = [1] * both_correct + [0] * only_a + [1] * only_b + [0] * both_wrong
    return a, b


INDEPENDENT = compare(A, B)
SHARED_SEEDS = compare(PAIRED_A, PAIRED_B, paired=True)
PER_EXAMPLE = compare_examples(*answers(800, 35, 50, 115))


def made_up(difference, ci_low, ci_high, p_value, confidence=0.95):
    """A stand-in result with exactly the numbers a test needs."""
    return SimpleNamespace(
        difference=difference, ci_low=ci_low, ci_high=ci_high,
        p_value=p_value, confidence=confidence,
    )


# --- The verdict -------------------------------------------------------------


def test_the_three_examples_get_the_three_verdicts():
    assert judge(INDEPENDENT, "A", "B").kind == "real"
    assert judge(SHARED_SEEDS, "A", "B").kind == "borderline"
    assert judge(PER_EXAMPLE, "A", "B").kind == "none"


def test_real_verdict_names_the_winner():
    assert judge(INDEPENDENT, "Baseline", "Ours").title == (
        "Likely real: Ours is better than Baseline"
    )


def test_winner_flips_when_lower_is_better():
    verdict = judge(INDEPENDENT, "Baseline", "Ours", higher_is_better=False)
    assert verdict.title == "Likely real: Baseline is better than Ours"


def test_winner_is_a_when_b_scores_lower():
    assert "A is better than B" in judge(compare(B, A), "A", "B").title


def test_p_between_the_bar_and_twice_the_bar_is_borderline():
    assert judge(made_up(0.5, -0.1, 1.1, p_value=0.07), "A", "B").kind == "borderline"
    assert judge(made_up(0.5, -0.1, 1.1, p_value=0.11), "A", "B").kind == "none"


def test_checks_that_disagree_are_borderline():
    p_passes_only = judge(made_up(0.5, -0.1, 1.1, p_value=0.04), "A", "B")
    interval_passes_only = judge(made_up(0.5, 0.1, 0.9, p_value=0.30), "A", "B")
    assert p_passes_only.kind == interval_passes_only.kind == "borderline"
    assert "disagree" in p_passes_only.reason
    assert "disagree" in interval_passes_only.reason


def test_the_bar_follows_the_confidence_level():
    # p = 0.03 passes at 95% confidence (bar 0.05) but not at 99% (bar 0.01).
    assert judge(made_up(0.5, 0.1, 0.9, 0.03, confidence=0.95), "A", "B").kind == "real"
    assert judge(made_up(0.5, 0.1, 0.9, 0.03, confidence=0.99), "A", "B").kind == "borderline"


def test_no_evidence_does_not_claim_the_models_are_equal():
    verdict = judge(PER_EXAMPLE, "A", "B")
    assert verdict.title == "No evidence of a difference"
    assert "does not prove" in verdict.reason


def test_borderline_says_what_to_collect_more_of():
    assert "More runs" in judge(SHARED_SEEDS, "A", "B").reason
    more = judge(made_up(1.0, -0.1, 2.1, 0.07), "A", "B", more_of="test examples")
    assert "More test examples" in more.reason


# --- Cautions ----------------------------------------------------------------


def test_good_data_has_no_cautions():
    assert seed_cautions(INDEPENDENT, "A", "B") == []
    assert example_cautions(PER_EXAMPLE) == []


def test_five_paired_runs_cannot_pass_the_bar():
    (caution,) = seed_cautions(SHARED_SEEDS, "A", "B")
    assert "5 paired runs" in caution
    assert "0.0625" in caution
    assert "at least 6 runs" in caution


def test_three_independent_runs_get_two_cautions():
    cautions = seed_cautions(compare(A[:3], B[:3]), "A", "B")
    assert len(cautions) == 2
    assert "3 and 3 runs" in cautions[0] and "at least 4 runs" in cautions[0]
    assert "fewer than 5 runs" in cautions[1]


def test_identical_scores_are_flagged():
    cautions = seed_cautions(compare([83.0] * 5, B), "Baseline", "Ours")
    assert any("All runs of Baseline" in caution for caution in cautions)


def test_small_test_set_and_few_disagreements_are_flagged():
    cautions = example_cautions(compare_examples(*answers(40, 3, 5, 2)))
    assert len(cautions) == 2
    assert "only 50 examples" in cautions[0]
    assert "only 8 examples" in cautions[1]


# --- The sentence for a paper ------------------------------------------------


def test_sentence_for_a_real_difference():
    sentence = seed_sentence(INDEPENDENT, judge(INDEPENDENT, "A", "B"), "A", "B")
    assert sentence == (
        "B outperforms A by 0.66 on average (95% bootstrap CI [0.28, 1.04]; "
        "two-sided permutation test, p = 0.0317; n = 5 runs per model)."
    )


def test_sentence_for_a_borderline_paired_difference():
    verdict = judge(SHARED_SEEDS, "A", "B")
    assert seed_sentence(SHARED_SEEDS, verdict, "A", "B") == (
        "The difference between B and A (+0.60 on average) is borderline "
        "(95% bootstrap CI [0.48, 0.72]; two-sided paired permutation test, "
        "p = 0.0625; n = 5 paired runs)."
    )


def test_sentence_when_a_wins_reports_a_positive_gap():
    result = compare(B, A)  # now "B" is the weaker model
    sentence = seed_sentence(result, judge(result, "Ours", "Baseline"), "Ours", "Baseline")
    assert sentence.startswith("Ours outperforms Baseline by 0.66 on average")
    # The interval for "B minus A" is negative here; the sentence flips it so
    # that it describes the winner's lead as positive numbers.
    assert result.ci_high < 0
    assert f"[{-result.ci_high:.2f}, {-result.ci_low:.2f}]" in sentence


def test_sentence_with_different_numbers_of_runs():
    result = compare(A, B[:4])
    assert "n = 5 and 4 runs" in seed_sentence(result, judge(result, "A", "B"), "A", "B")


def test_sentence_uses_the_requested_decimals():
    a = [score / 100 for score in A]
    b = [score / 100 for score in B]
    result = compare(a, b)
    sentence = seed_sentence(result, judge(result, "A", "B"), "A", "B", decimals=4)
    assert "by 0.0066 on average" in sentence


def test_sentence_for_per_example_results():
    verdict = judge(PER_EXAMPLE, "A", "B")
    sentence = example_sentence(PER_EXAMPLE, verdict, "A", "B")
    assert sentence.startswith(
        "The difference between B and A (+1.50 percentage points) is not "
        "statistically significant (95% bootstrap CI ["
    )
    assert sentence.endswith("McNemar's exact test, p = 0.128; n = 1,000 test examples).")


def test_sentence_for_a_real_per_example_difference():
    result = compare_examples(*answers(800, 20, 80, 100))
    sentence = example_sentence(result, judge(result, "A", "B"), "A", "B")
    assert sentence.startswith("B outperforms A by 6.00 percentage points (95% bootstrap CI [")
    assert "p < 0.001" in sentence


def test_p_value_formatting():
    assert format_p(8 / 252) == "0.0317"
    assert format_p(0.0625) == "0.0625"
    assert format_p(0.00004) == "< 0.001"
    assert format_p(1.0) == "1"
