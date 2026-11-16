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
