"""easyCI - is the difference between two models real, or just luck?

This file is the screen of the app. Run it with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="easyCI", page_icon="📊")

st.title("easyCI")
st.subheader("Is your model's improvement real, or just luck?")

st.write(
    "Give easyCI the scores of two models. It tells you whether the "
    "difference is statistically real, with a confidence interval, a plot "
    "and a sentence you can paste into a paper."
)

st.info("Setup works. The analysis tools arrive in the next steps.")
