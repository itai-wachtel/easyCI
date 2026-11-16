"""easyCI - is the difference between two models real, or just luck?

This file is the screen of the app. It contains no maths: it collects the
inputs, calls compare() from stats/seeds.py, and shows the results.

Run it with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from parsing import parse_scores, scores_from_csv
from plots import difference_plot, runs_plot
from stats.seeds import compare

EXAMPLES = {
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


def load_example(name):
    """Fill the input boxes with one of the built-in examples."""
    st.session_state.update(EXAMPLES[name])


def load_csv():
    """Fill the input boxes from an uploaded CSV file."""
    file = st.session_state.csv_file
    st.session_state.csv_error = None
    if file is None:
        return
    try:
        name_a, a, name_b, b = scores_from_csv(file)
    except Exception as error:  # unreadable file, or not two number columns
        st.session_state.csv_error = str(error)
        return
    st.session_state.name_a = name_a
    st.session_state.name_b = name_b
    st.session_state.text_a = ", ".join(str(score) for score in a)
    st.session_state.text_b = ", ".join(str(score) for score in b)


@st.cache_data
def run_compare(a, b, paired, higher_is_better, confidence):
    """Call compare(). Streamlit remembers the answer until the inputs change."""
    return compare(a, b, paired, higher_is_better, confidence)


def format_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def format_d(d):
    return "∞" if abs(d) == float("inf") else f"{d:.2f}"


st.set_page_config(page_title="easyCI", page_icon="📊")

st.title("easyCI")
st.subheader("Is your model's improvement real, or just luck?")
st.write(
    "Train each model several times with different seeds, paste one score per "
    "run, and easyCI tells you how much of the difference you can trust."
)

# ---------------------------------------------------------------- Inputs ---

st.header("Scores", divider="gray")

example_1, example_2, _ = st.columns([1.4, 1.4, 2])
example_1.button(
    "Example: independent runs", on_click=load_example, args=["independent"], width="stretch"
)
example_2.button(
    "Example: shared seeds", on_click=load_example, args=["shared seeds"], width="stretch"
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
with right:
    name_b = st.text_input("Name of the second model", key="name_b") or "Model B"
    text_b = st.text_area(
        f"Scores of {name_b}", key="text_b", height=110, placeholder="84.2, 83.8, 84.5, ..."
    )
st.caption("One score per training run, separated by commas, spaces or new lines.")

with st.expander("Or upload a CSV file"):
    st.file_uploader(
        "A file with one column of scores per model. The column titles become the names.",
        type="csv",
        key="csv_file",
        on_change=load_csv,
    )
    if st.session_state.get("csv_error"):
        st.error(st.session_state.csv_error)

# -------------------------------------------------------------- Settings ---

st.header("Settings", divider="gray")

paired = st.checkbox(
    "Both models used the same seeds (paired)",
    key="paired",
    help="Tick this if run 1 of both models shared a seed, run 2 shared a seed, and so on.",
)
setting_1, setting_2 = st.columns(2)
direction = setting_1.radio(
    "Which score is better?", ["Higher is better", "Lower is better"], horizontal=True
)
confidence = setting_2.selectbox(
    "Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda c: f"{c:.0%}"
)
higher_is_better = direction == "Higher is better"

# --------------------------------------------------------------- Results ---

st.header("Results", divider="gray")

try:
    a = parse_scores(text_a)
    b = parse_scores(text_b)
    if not a or not b:
        st.info("Paste the scores of both models above, or load an example.")
        st.stop()
    # Tuples, because Streamlit can only remember results for unchangeable inputs.
    result = run_compare(tuple(a), tuple(b), paired, higher_is_better, confidence)
except ValueError as error:
    st.error(str(error))
    st.stop()

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
    st.metric(f"Difference ({name_b} − {name_a})", f"{result.difference:+.2f}")
    st.caption(direction_text)
with second:
    st.metric(f"{confidence:.0%} confidence interval", f"{result.ci_low:.2f} to {result.ci_high:.2f}")
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
    st.caption(
        "Of the seeds." if paired else "Of all match-ups of one run against one run."
    )

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
        "Mean": st.column_config.NumberColumn(format="%.2f"),
        "Standard deviation": st.column_config.NumberColumn(format="%.2f"),
    },
)

# ----------------------------------------------------------------- Plots ---

dark = st.context.theme.type == "dark"

st.subheader("The runs")
st.caption(
    "One dot per run. Lines join the runs that shared a seed."
    if paired
    else "One dot per run. The short line is each model's mean."
)
st.plotly_chart(runs_plot(a, b, name_a, name_b, paired, dark), theme=None)

st.subheader("The difference")
st.caption(
    f"The bar is the {confidence:.0%} confidence interval. "
    "If it does not touch the dashed line, the difference is plausibly real."
)
st.plotly_chart(difference_plot(result, name_a, name_b, dark), theme=None)
