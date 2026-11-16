"""Turn what the user typed or uploaded into lists of numbers."""

import re

import pandas as pd

# The ways a CSV file may write "correct" and "wrong".
ANSWER_WORDS = {
    "1": True, "1.0": True, "true": True, "yes": True,
    "0": False, "0.0": False, "false": False, "no": False,
}


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


def _read_csv(file):
    """Open a CSV file as a table, with a clear error if that fails."""
    # The page reads an uploaded file again on every click, so always go back
    # to the start of the file first.
    if hasattr(file, "seek"):
        file.seek(0)
    try:
        return pd.read_csv(file)
    except Exception:
        raise ValueError("easyCI could not read this file. Is it a CSV file?") from None


def score_columns(file):
    """Read every column of numbers in a CSV file.

    Returns a dictionary {column title: list of scores}. Empty cells at the
    end of a column are dropped, so columns may differ in length. An empty
    cell in the middle is kept as "missing", so that it is reported instead of
    silently shifting the scores below it.
    """
    table = _read_csv(file)
    columns = {}
    for name in table.select_dtypes("number").columns:
        last_filled = table[name].last_valid_index()
        if last_filled is not None:  # skip columns that are completely empty
            columns[str(name)] = table[name].loc[:last_filled].tolist()
    if len(columns) < 2:
        raise ValueError("The CSV file needs two columns of numbers, one per model.")
    return columns


def answer_columns(file):
    """Read every right/wrong column in a CSV file.

    Each row is one test example. A right/wrong column holds 1 (correct) or
    0 (wrong) in every row; true/false and yes/no also work.
    Returns a dictionary {column title: list of True/False}.
    """
    table = _read_csv(file)
    columns = {}
    for name in table.columns:
        words = table[name].astype(str).str.strip().str.lower()
        if len(table) > 0 and table[name].notna().all() and words.isin(ANSWER_WORDS).all():
            columns[str(name)] = words.map(ANSWER_WORDS).tolist()
    if len(columns) < 2:
        raise ValueError(
            "The CSV file needs two columns of 1 (correct) and 0 (wrong), one per "
            "model, with a value in every row."
        )
    return columns
