"""Checks of the page itself: open it, click and type like a user would.

Streamlit's AppTest runs app.py without a browser and lets the test press
buttons, fill boxes and read what the page shows.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).parent.parent / "app.py")
EXAMPLES_MODE = "One run per model, results per test example"
SEEDS_MODE = "Several runs per model"


def open_app():
    app = AppTest.from_file(APP_FILE, default_timeout=30).run()
    assert not app.exception
    return app


def click(app, label):
    """Press the button with this text."""
    button = next(button for button in app.button if button.label == label)
    button.click().run()
    assert not app.exception


def metrics(app):
    """The headline numbers on the page, as {title: value}."""
    return {metric.label: metric.value for metric in app.metric}


def errors(app):
    return [error.value for error in app.error]


# --- Mode 1: several runs per model ----------------------------------------


def test_page_opens_and_asks_for_scores():
    app = open_app()
    assert not app.metric
    assert "Paste the scores" in app.info[0].value


def test_independent_example_shows_the_known_results():
    app = open_app()
    click(app, "Example: independent runs")
    shown = metrics(app)
    assert shown["Difference (Model B − Model A)"] == "+0.66"
    assert shown["95% confidence interval"] == "0.28 to 1.04"
    assert shown["p-value"] == "0.0317"
    assert shown["Effect size (Cohen's d)"] == "1.91"
    assert shown["Model B is better"] == "92%"


def test_shared_seeds_example_ticks_paired():
    app = open_app()
    click(app, "Example: shared seeds")
    assert app.checkbox[0].value is True
    shown = metrics(app)
    assert shown["p-value"] == "0.0625"
    assert shown["Model B is better"] == "100%"


def test_unticking_paired_hides_the_effect():
    app = open_app()
    click(app, "Example: shared seeds")
    app.checkbox[0].uncheck().run()
    assert float(metrics(app)["p-value"]) > 0.5


def test_text_that_is_not_a_number_shows_an_error_and_no_results():
    app = open_app()
    click(app, "Example: independent runs")
    app.text_area[0].set_value("83.1, abc").run()
    assert not app.exception
    assert any('"abc" is not a number' in message for message in errors(app))
    assert not app.metric


def test_too_few_runs_shows_an_error():
    app = open_app()
    app.text_area[0].set_value("83.1")
    app.text_area[1].set_value("84.2, 83.8").run()
    assert any("at least 2 runs" in message for message in errors(app))


def test_paired_with_different_numbers_of_runs_shows_an_error():
    app = open_app()
    click(app, "Example: shared seeds")
    app.text_area[1].set_value("80.6, 85.5, 82.7").run()
    assert any("same number of runs" in message for message in errors(app))


def test_scores_between_0_and_1_are_not_rounded_away():
    app = open_app()
    app.text_area[0].set_value("0.831, 0.839, 0.835, 0.840, 0.832")
    app.text_area[1].set_value("0.842, 0.838, 0.845, 0.841, 0.844").run()
    shown = metrics(app)
    assert shown["Difference (Model B − Model A)"] == "+0.0066"
    assert shown["95% confidence interval"] == "0.0028 to 0.0104"


def test_lower_is_better_flips_who_wins():
    app = open_app()
    click(app, "Example: independent runs")
    app.radio[1].set_value("Lower is better").run()
    assert metrics(app)["Model B is better"] == "8%"


def test_model_names_are_used_in_the_results():
    app = open_app()
    click(app, "Example: independent runs")
    app.text_input[1].set_value("ResNet").run()
    assert "ResNet is better" in metrics(app)


def test_the_page_says_how_many_scores_it_read():
    app = open_app()
    click(app, "Example: independent runs")
    captions = [caption.value for caption in app.caption]
    assert captions.count("5 scores read.") == 2


# --- Mode 2: one run per model, per test example ---------------------------


def test_per_example_mode_shows_the_known_results():
    app = open_app()
    app.radio[0].set_value(EXAMPLES_MODE).run()
    click(app, "Example: 1,000 test examples")
    shown = metrics(app)
    assert shown["Accuracy of Model A"] == "83.5%"
    assert shown["Accuracy of Model B"] == "85.0%"
    assert shown["Difference (Model B − Model A)"] == "+1.50 points"
    assert shown["p-value (McNemar's test)"] == "0.128"
    assert shown["Disagreements"] == "85"


def test_changing_a_setting_keeps_the_per_example_results():
    app = open_app()
    app.radio[0].set_value(EXAMPLES_MODE).run()
    click(app, "Example: 1,000 test examples")
    app.selectbox[0].set_value(0.99).run()
    assert not app.exception
    assert "99% confidence interval" in metrics(app)


# --- Switching between the modes -------------------------------------------


def test_switching_modes_keeps_what_was_typed():
    app = open_app()
    click(app, "Example: shared seeds")
    app.radio[0].set_value(EXAMPLES_MODE).run()
    app.radio[0].set_value(SEEDS_MODE).run()
    assert app.text_area[0].value == "80.0, 85.0, 82.0, 88.0, 84.0"
    assert app.checkbox[0].value is True
    assert metrics(app)["p-value"] == "0.0625"
