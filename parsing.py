"""Turn what the user typed or uploaded into lists of numbers."""

import re

import pandas as pd


def parse_scores(text):
    """Turn text like "83.1, 83.9 83.5" into [83.1, 83.9, 83.5].

    Numbers can be separated by commas, semicolons, spaces or new lines.
    A percent sign after a number is ignored.
    """
    pieces = [piece for piece in re.split(r"[,;\s]+", text.strip()) if piece]
    scores = []
    for piece in pieces:
        try:
            scores.append(float(piece.rstrip("%")))
        except ValueError:
            raise ValueError(f'"{piece}" is not a number.') from None
    return scores


# The ways a CSV file may write "correct" and "wrong".
ANSWER_WORDS = {
    "1": True, "1.0": True, "true": True, "yes": True,
    "0": False, "0.0": False, "false": False, "no": False,
}


def _as_answers(column):
    """Return a CSV column as a list of True/False, or None if it is not one."""
    words = column.astype(str).str.strip().str.lower()
    if column.isna().any() or not words.isin(ANSWER_WORDS).all():
        return None
    return words.map(ANSWER_WORDS).tolist()


def answers_from_csv(file):
    """Read the first two right/wrong columns of a CSV file.

    Each row is one test example; each column holds 1 (correct) or 0 (wrong)
    for one model. Returns (name_a, answers_a, name_b, answers_b).
    """
    table = pd.read_csv(file)
    found = []
    for name in table.columns:
        answers = _as_answers(table[name])
        if answers is not None:
            found.append((str(name), answers))
    if len(found) < 2 or not found[0][1]:
        raise ValueError(
            "The CSV file needs two columns of 1 (correct) and 0 (wrong), one per "
            "model, with a value in every row."
        )
    (name_a, a), (name_b, b) = found[:2]
    return name_a, a, name_b, b


def scores_from_csv(file):
    """Read the first two number columns of a CSV file.

    Returns (name_a, scores_a, name_b, scores_b). The column titles become the
    model names. Empty cells are skipped, so the columns may differ in length.
    """
    table = pd.read_csv(file)
    numeric = table.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise ValueError("The CSV file needs two columns of numbers, one per model.")
    name_a, name_b = numeric.columns[:2]
    return (
        str(name_a),
        numeric[name_a].dropna().tolist(),
        str(name_b),
        numeric[name_b].dropna().tolist(),
    )
