"""easyCI - is the difference between two models real, or just luck?

This file is the screen of the app. It contains no maths: it collects the
inputs, calls the functions in the stats folder, and shows the results.

The app has two modes, each drawn by its own function:
    seeds_screen()     several training runs per model, one score per run
    examples_screen()  one run per model, right/wrong for every test example

Run it with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from parsing import answer_columns, parse_scores, score_columns
from plots import accuracy_plot, difference_plot, runs_plot
from stats.examples import compare_examples
from stats.planning import runs_needed
from stats.seeds import compare
from stats.verdict import (
    MULTIPLE_COMPARISONS_NOTE,
    example_cautions,
    example_sentence,
    format_p,
    judge,
    seed_cautions,
    seed_sentence,
)

SEEDS_MODE = "Several runs per model"
EXAMPLES_MODE = "One run per model, results per test example"

SEED_EXAMPLES = {
    "independent": {
        "text_a": "83.1, 83.9, 83.5, 84.0, 83.2",
        "text_b": "84.2, 83.8, 84.5, 84.1, 84.4",
        "paired": False,
    },
    "shared seeds": {
        "text_a": "80.0, 85.0, 82.0, 88.0, 84.0",
        "text_b": "80.6, 85.5, 82.7, 88.4, 84.8",
        "paired": True,
    },
}
SAMPLE_ANSWERS_FILE = Path(__file__).parent / "sample_data" / "per_example.csv"


# --------------------------------------------------------------- Helpers ---


def load_seed_example(name):
    """Fill the score boxes with one of the built-in examples."""
    st.session_state.update(SEED_EXAMPLES[name])


def load_scores_csv(new_file=False):
    """Fill the score boxes from two columns of the uploaded CSV file.

    Runs when a file is uploaded (new_file=True: start with the first two
    columns) and when the user picks a different column in one of the menus.
    """
    file = st.session_state.csv_file
    st.session_state.csv_error = None
    if file is None:
        return
    try:
        columns = score_columns(file)
    except ValueError as error:
        st.session_state.csv_error = str(error)
        return
    titles = list(columns)
    name_a = st.session_state.get("csv_column_a")
    name_b = st.session_state.get("csv_column_b")
    if new_file or name_a not in columns or name_b not in columns:
        name_a, name_b = titles[0], titles[1]
    st.session_state.csv_column_a = name_a
    st.session_state.csv_column_b = name_b
    st.session_state.name_a = name_a
    st.session_state.name_b = name_b
    st.session_state.text_a = ", ".join(str(score) for score in columns[name_a])
    st.session_state.text_b = ", ".join(str(score) for score in columns[name_b])


def read_scores(text, name):
    """Turn one score box into numbers and say under the box what was read.

    Returns None (after showing the error) if the text cannot be read.
    """
    try:
        scores = parse_scores(text)
    except ValueError as error:
        st.error(f"{name}: {error}")
        return None
    if scores:
        st.caption(f"{len(scores)} scores read.")
    return scores


def use_sample_answers():
    """Remember that the user asked for the built-in per-example file."""
    st.session_state.use_sample_answers = True


# Streamlit reruns this whole file on every click. @st.cache_data makes it
# remember a result until the inputs change. The inputs are tuples because
# Streamlit can only remember results for unchangeable inputs.


@st.cache_data
def run_compare(a, b, paired, higher_is_better, confidence):
    return compare(a, b, paired, higher_is_better, confidence)


@st.cache_data
def run_compare_examples(a, b, confidence):
    return compare_examples(a, b, confidence)


def show_verdict(verdict, cautions):
    """Draw the verdict as a coloured box, with any cautions listed under it."""
    box = {"real": st.success, "borderline": st.warning, "none": st.info}[verdict.kind]
    box(f"**{verdict.title}**\n\n{verdict.reason}")
    if cautions:
        st.markdown(
            "**Before you rely on this:**\n" + "\n".join(f"- {caution}" for caution in cautions)
        )


def show_sentence(sentence):
    """Draw the sentence for a paper in a box that has a copy button."""
    st.subheader("Sentence for your paper")
    st.code(sentence, language=None, wrap_lines=True)
    st.caption(MULTIPLE_COMPARISONS_NOTE)


def show_runs_calculator(result, digits):
    """Draw the "how many runs do you need?" calculator of the per-seed mode."""
    st.subheader("How many runs do you need?")
    st.write(
        "Enter the smallest difference you would care about. easyCI estimates how "
        "many runs it takes to detect a true difference of that size."
    )
    target = st.number_input(
        "Smallest difference worth detecting",
        min_value=0.0,
        value=round(abs(result.difference), digits),  # start at the observed one
        step=10.0**-digits,
        format=f"%.{digits}f",
    )
    needed = runs_needed(target, result.noise, result.paired, result.confidence)
    if needed is None:
        st.info("Enter a difference above zero.")
        return
    st.metric("Runs needed", f"{needed} paired runs" if result.paired else f"{needed} per model")
    st.caption(
        f"For an 80% chance of passing the {1 - result.confidence:.2f} bar, if the "
        f"run-to-run noise stays at {result.noise:.{digits}f}. This is an "
        f"approximation, and the noise is itself estimated from only "
        f"{result.summary_a.n} runs, so treat the answer as a guide."
    )
    if needed < 10:
        st.caption(
            "Known limitation: when this recommends fewer than about 10 runs it is "
            "optimistic. In easyCI's validation the real chance was 66% to 72%, not "
            "80%, so add a run or two."
        )


def decimals_for(scores):
    """How many decimals to show: more when the scores are small numbers.

    Scores like 83.5 need two decimals, but scores like 0.835 need four,
    otherwise a difference of 0.007 would be shown as 0.01.
    """
    largest = max(abs(score) for score in scores)
    if largest >= 10:
        return 2
    if largest >= 1:
        return 3
    return 4


def format_d(d):
    return "∞" if abs(d) == float("inf") else f"{d:.2f}"


def confidence_box(key):
    return st.selectbox(
        "Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda c: f"{c:.0%}", key=key
    )


def is_dark():
    return st.context.theme.type == "dark"


# ------------------------------------------- Mode 1: several runs per model ---


def seeds_screen():
    st.header("Scores", divider="gray")

    example_1, example_2, _ = st.columns([1.4, 1.4, 2])
    example_1.button(
        "Example: independent runs",
        on_click=load_seed_example,
        args=["independent"],
        width="stretch",
    )
    example_2.button(
        "Example: shared seeds", on_click=load_seed_example, args=["shared seeds"], width="stretch"
    )

    # Starting values of the name boxes (only set the first time the page runs).
    st.session_state.setdefault("name_a", "Model A")
    st.session_state.setdefault("name_b", "Model B")

    left, right = st.columns(2)
    with left:
        name_a = st.text_input("Name of the first model", key="name_a") or "Model A"
        text_a = st.text_area(
            f"Scores of {name_a}", key="text_a", height=110, placeholder="83.1, 83.9, 83.5, ..."
        )
        a = read_scores(text_a, name_a)
    with right:
        name_b = st.text_input("Name of the second model", key="name_b") or "Model B"
        text_b = st.text_area(
            f"Scores of {name_b}", key="text_b", height=110, placeholder="84.2, 83.8, 84.5, ..."
        )
        b = read_scores(text_b, name_b)
    st.caption(
        "One score per training run, separated by commas, spaces or new lines. "
        "Write decimals with a point: 83.1, not 83,1."
    )

    with st.expander("Or upload a CSV file"):
        file = st.file_uploader(
            "A file with one column of scores per model. The column titles become the names.",
            type="csv",
            key="csv_file",
            on_change=load_scores_csv,
            args=[True],
        )
        if st.session_state.get("csv_error"):
            st.error(st.session_state.csv_error)
        elif file is not None:
            # Let the user say which column is which model. This matters when
            # the file has other number columns, such as the seed.
            titles = list(score_columns(file))
            pick_a, pick_b = st.columns(2)
            pick_a.selectbox(
                "Column of the first model", titles, key="csv_column_a", on_change=load_scores_csv
            )
            pick_b.selectbox(
                "Column of the second model", titles, key="csv_column_b", on_change=load_scores_csv
            )

    st.header("Settings", divider="gray")

    paired = st.checkbox(
        "Both models used the same seeds (paired)",
        key="paired",
        help="Tick this if run 1 of both models shared a seed, run 2 shared a seed, and so on.",
    )
    setting_1, setting_2 = st.columns(2)
    with setting_1:
        direction = st.radio(
            "Which score is better?", ["Higher is better", "Lower is better"], horizontal=True
        )
    with setting_2:
        confidence = confidence_box("seeds_confidence")
    higher_is_better = direction == "Higher is better"

    st.header("Results", divider="gray")

    if a is None or b is None:
        st.info("Fix the scores marked in red above to see the results.")
        return
    if not a or not b:
        st.info("Paste the scores of both models above, or load an example.")
        return
    try:
        result = run_compare(tuple(a), tuple(b), paired, higher_is_better, confidence)
    except ValueError as error:
        st.error(str(error))
        return
    digits = decimals_for(a + b)

    verdict = judge(result, name_a, name_b, higher_is_better)
    show_verdict(verdict, seed_cautions(result, name_a, name_b))

    if result.difference == 0:
        direction_text = f"{name_b} and {name_a} have the same average."
    else:
        word = "higher" if result.difference > 0 else "lower"
        b_is_better = (result.difference > 0) == higher_is_better
        direction_text = (
            f"{name_b} is {word} on average, which is {'better' if b_is_better else 'worse'}."
        )

    first, second, third = st.columns(3)
    with first:
        st.metric(f"Difference ({name_b} − {name_a})", f"{result.difference:+.{digits}f}")
        st.caption(direction_text)
    with second:
        st.metric(
            f"{confidence:.0%} confidence interval",
            f"{result.ci_low:.{digits}f} to {result.ci_high:.{digits}f}",
        )
        st.caption("The plausible range of the true difference. If it includes 0, be careful.")
    with third:
        st.metric("p-value", format_p(result.p_value))
        st.caption("How often luck alone gives a gap this big. Below 0.05 is the usual bar.")

    fourth, fifth, _ = st.columns(3)
    with fourth:
        st.metric("Effect size (Cohen's d)", format_d(result.cohens_d))
        st.caption("The gap in units of run-to-run noise. Around 0.8 or more is large.")
    with fifth:
        st.metric(f"{name_b} is better", f"{result.superiority:.0%}")
        st.caption("Of the seeds." if paired else "Of all match-ups of one run against one run.")

    st.dataframe(
        pd.DataFrame(
            {
                "Model": [name_a, name_b],
                "Mean": [result.summary_a.mean, result.summary_b.mean],
                "Standard deviation": [result.summary_a.std, result.summary_b.std],
                "Runs": [result.summary_a.n, result.summary_b.n],
            }
        ),
        hide_index=True,
        column_config={
            "Mean": st.column_config.NumberColumn(format=f"%.{digits}f"),
            "Standard deviation": st.column_config.NumberColumn(format=f"%.{digits}f"),
        },
    )

    show_sentence(seed_sentence(result, verdict, name_a, name_b, digits))

    st.subheader("The runs")
    st.caption(
        "One dot per run. Lines join the runs that shared a seed."
        if paired
        else "One dot per run. The short line is each model's mean."
    )
    st.plotly_chart(runs_plot(a, b, name_a, name_b, paired, is_dark(), digits), theme=None)

    st.subheader("The difference")
    st.caption(
        f"The bar is the {confidence:.0%} confidence interval. "
        "If it does not touch the dashed line, the difference is plausibly real."
    )
    st.plotly_chart(
        difference_plot(result, name_a, name_b, is_dark(), decimals=digits), theme=None
    )

    show_runs_calculator(result, digits)


# ------------------------------ Mode 2: one run per model, per test example ---


def examples_screen():
    st.header("Results per test example", divider="gray")
    st.write(
        "Upload a CSV file with one row per test example and one column per model: "
        "1 if the model answered that example correctly, 0 if not. "
        "The column titles become the model names."
    )
    st.button("Example: 1,000 test examples", on_click=use_sample_answers)
    upload = st.file_uploader("CSV file", type="csv", key="answers_file")
    confidence = confidence_box("examples_confidence")

    st.header("Results", divider="gray")

    if upload is not None:
        source = upload
    elif st.session_state.get("use_sample_answers"):
        source = SAMPLE_ANSWERS_FILE
    else:
        st.info("Upload a CSV file above, or load the example.")
        return

    try:
        columns = answer_columns(source)
    except ValueError as error:
        st.error(str(error))
        return
    titles = list(columns)
    name_a, name_b = titles[0], titles[1]
    if len(titles) > 2:
        # More than two right/wrong columns (one may be the true label, for
        # example), so the user has to say which two are the models.
        pick_a, pick_b = st.columns(2)
        name_a = pick_a.selectbox("Column of the first model", titles, index=0)
        name_b = pick_b.selectbox("Column of the second model", titles, index=1)
    result = run_compare_examples(tuple(columns[name_a]), tuple(columns[name_b]), confidence)
    table = result.table

    verdict = judge(result, name_a, name_b, more_of="test examples")
    show_verdict(verdict, example_cautions(result))

    first, second, third = st.columns(3)
    with first:
        st.metric(f"Accuracy of {name_a}", f"{result.accuracy_a:.1f}%")
        st.caption(
            f"{confidence:.0%} interval: "
            f"{result.interval_a[0]:.1f}% to {result.interval_a[1]:.1f}%"
        )
    with second:
        st.metric(f"Accuracy of {name_b}", f"{result.accuracy_b:.1f}%")
        st.caption(
            f"{confidence:.0%} interval: "
            f"{result.interval_b[0]:.1f}% to {result.interval_b[1]:.1f}%"
        )
    with third:
        st.metric(f"Difference ({name_b} − {name_a})", f"{result.difference:+.2f} points")
        st.caption(f"Measured on {table.n:,} test examples.")

    fourth, fifth, sixth = st.columns(3)
    with fourth:
        st.metric(
            f"{confidence:.0%} confidence interval",
            f"{result.ci_low:.2f} to {result.ci_high:.2f}",
        )
        st.caption("The plausible range of the true difference, in points.")
    with fifth:
        st.metric("p-value (McNemar's test)", format_p(result.p_value))
        st.caption("How often luck alone gives a split this uneven. Below 0.05 is the usual bar.")
    with sixth:
        st.metric("Disagreements", f"{table.only_a + table.only_b:,}")
        st.caption(f"{name_a} won {table.only_a:,} of them and {name_b} won {table.only_b:,}.")

    st.subheader("Where the models agree and disagree")
    st.caption(
        "Number of test examples in each group. Only the two disagreement "
        "groups can show which model is better."
    )
    st.dataframe(
        pd.DataFrame(
            {
                f"{name_b} correct": [table.both_correct, table.only_b],
                f"{name_b} wrong": [table.only_a, table.both_wrong],
            },
            index=[f"{name_a} correct", f"{name_a} wrong"],
        )
    )

    st.info(
        "This compares the two trained models you tested, not their training "
        "methods: with one run per model, luck in the random seed is invisible. "
        f"To compare methods, train each one several times and use “{SEEDS_MODE}”."
    )

    show_sentence(example_sentence(result, verdict, name_a, name_b))

    st.subheader("The accuracies")
    st.caption(f"Each bar is the {confidence:.0%} interval of one model's accuracy.")
    st.plotly_chart(accuracy_plot(result, name_a, name_b, is_dark()), theme=None)

    st.subheader("The difference")
    st.caption(
        f"The bar is the {confidence:.0%} confidence interval. "
        "If it does not touch the dashed line, the difference is plausibly real."
    )
    st.plotly_chart(
        difference_plot(result, name_a, name_b, is_dark(), unit=" (percentage points)"),
        theme=None,
    )


# -------------------------------------------------------------- The page ---

st.set_page_config(page_title="easyCI", page_icon="📊")

st.title("easyCI")
st.subheader("Is your model's improvement real, or just luck?")
st.write(
    "Give easyCI the results of two models, and it tells you how much of the "
    "difference between them you can trust."
)

# Streamlit forgets an input box once it is off screen. Saving each value
# again keeps what the user typed when they switch between the two modes.
for key in ["name_a", "name_b", "text_a", "text_b", "paired", "seeds_confidence", "examples_confidence"]:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

mode = st.radio("What results do you have?", [SEEDS_MODE, EXAMPLES_MODE], horizontal=True)

if mode == SEEDS_MODE:
    seeds_screen()
else:
    examples_screen()
