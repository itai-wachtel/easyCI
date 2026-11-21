"""Checks that pasted text and CSV files are read correctly."""

import io
import math
from pathlib import Path

import pytest

from parsing import (
    answer_columns,
    default_columns,
    looks_like_index,
    parse_scores,
    score_columns,
)

# --- Pasted text -------------------------------------------------------------


def test_commas():
    assert parse_scores("83.1, 83.9, 83.5") == [83.1, 83.9, 83.5]


def test_spaces_and_new_lines():
    assert parse_scores("83.1 83.9\n83.5\n") == [83.1, 83.9, 83.5]


def test_mixed_separators_and_extra_commas():
    assert parse_scores(" 83.1,83.9;  83.5, ") == [83.1, 83.9, 83.5]


def test_percent_signs_are_ignored():
    assert parse_scores("83.1%, 83.9%") == [83.1, 83.9]


def test_negative_and_whole_numbers():
    assert parse_scores("-0.5 2") == [-0.5, 2.0]


def test_empty_text_gives_empty_list():
    assert parse_scores("") == []
    assert parse_scores("   \n ") == []


def test_bad_text_names_the_problem():
    with pytest.raises(ValueError, match='"abc" is not a number'):
        parse_scores("83.1, abc")


# --- CSV files of scores -----------------------------------------------------


def test_score_columns_reads_each_model():
    file = io.StringIO("baseline,ours\n83.1,84.2\n83.9,83.8\n")
    assert score_columns(file) == {"baseline": [83.1, 83.9], "ours": [84.2, 83.8]}


def test_score_columns_returns_every_number_column_so_the_user_can_choose():
    file = io.StringIO("seed,baseline,ours\n1,83.1,84.2\n2,83.9,83.8\n")
    assert list(score_columns(file)) == ["seed", "baseline", "ours"]


def test_score_columns_can_differ_in_length():
    columns = score_columns(io.StringIO("baseline,ours\n83.1,84.2\n83.9,\n"))
    assert columns == {"baseline": [83.1, 83.9], "ours": [84.2]}


def test_empty_cell_in_the_middle_is_kept_as_missing():
    # Dropping it would shift the scores below and break the pairing by seed.
    columns = score_columns(io.StringIO("a,b\n80,80.6\n,85.5\n82,82.7\n"))
    assert columns["a"][0] == 80 and columns["a"][2] == 82
    assert math.isnan(columns["a"][1])


def test_score_columns_skips_text_columns():
    file = io.StringIO("run_name,baseline,ours\nfirst,83.1,84.2\nsecond,83.9,83.8\n")
    assert list(score_columns(file)) == ["baseline", "ours"]


def test_score_csv_with_one_column_is_rejected():
    with pytest.raises(ValueError, match="two columns"):
        score_columns(io.StringIO("baseline\n83.1\n83.9\n"))


# --- Which two columns to start with -----------------------------------------


@pytest.mark.parametrize("title", ["seed", "Seed", "run_id", "Run", "ID", "Unnamed: 0", "epoch"])
def test_titles_that_number_the_runs(title):
    assert looks_like_index(title, [42.0, 7.0, 123.0])


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_values_that_count_up_by_one(values):
    assert looks_like_index("x", values)


@pytest.mark.parametrize(
    "title, values",
    [
        ("baseline", [83.1, 83.9, 83.5]),
        ("ours", [80.0, 81.0, 82.0]),  # counts up by one, but not from 0 or 1
        ("ours", [1.0, 2.0]),  # too short to tell
        ("loss", [1.0, float("nan"), 3.0]),
        ("runtime", [1.0, 2.0, 3.0, 5.0]),  # "runtime" is not "run"
    ],
)
def test_score_columns_are_not_mistaken_for_numbering(title, values):
    assert not looks_like_index(title, values)


def test_default_columns_skip_the_seed():
    columns = {"seed": [1, 2, 3], "baseline": [83.1, 83.9, 83.5], "ours": [84.2, 83.8, 84.5]}
    assert default_columns(columns) == ("baseline", "ours")


def test_default_columns_fall_back_to_the_first_two():
    # Only one column looks like scores, so there is nothing better to offer.
    columns = {"seed": [1, 2, 3], "ours": [84.2, 83.8, 84.5]}
    assert default_columns(columns) == ("seed", "ours")


# --- CSV files of right/wrong answers ----------------------------------------


def test_answer_columns_with_ones_and_zeros():
    file = io.StringIO("baseline,ours\n1,1\n0,1\n1,0\n")
    assert answer_columns(file) == {
        "baseline": [True, False, True],
        "ours": [True, True, False],
    }


def test_answer_columns_with_true_false_words():
    file = io.StringIO("baseline,ours\nTrue,false\nFALSE,true\n")
    assert answer_columns(file) == {"baseline": [True, False], "ours": [False, True]}


def test_answer_columns_returns_every_candidate_so_the_user_can_choose():
    file = io.StringIO("example_id,label,baseline,ours\n17,1,1,0\n18,0,0,0\n")
    assert list(answer_columns(file)) == ["label", "baseline", "ours"]


def test_answer_csv_with_a_missing_value_is_rejected():
    with pytest.raises(ValueError, match="two columns of 1"):
        answer_columns(io.StringIO("baseline,ours\n1,1\n0,\n"))


def test_answer_csv_with_scores_instead_of_answers_is_rejected():
    with pytest.raises(ValueError, match="two columns of 1"):
        answer_columns(io.StringIO("baseline,ours\n83.1,84.2\n83.9,83.8\n"))


# --- The sample files that ship with easyCI ----------------------------------

SAMPLES = Path(__file__).parent.parent / "sample_data"


def test_per_seed_sample_file():
    columns = score_columns(SAMPLES / "per_seed.csv")
    assert list(columns) == ["seed", "Baseline", "New model"]
    assert len(columns["Baseline"]) == len(columns["New model"]) == 6
    # The page must start with the two models, not with the seed column.
    assert default_columns(columns) == ("Baseline", "New model")


def test_per_example_sample_file():
    columns = answer_columns(SAMPLES / "per_example.csv")
    assert list(columns) == ["Model A", "Model B"]
    assert sum(columns["Model A"]) == 835 and sum(columns["Model B"]) == 850


# --- Files in general --------------------------------------------------------


def test_the_same_file_can_be_read_again():
    # The page reads the uploaded file again every time the user clicks.
    file = io.BytesIO(b"baseline,ours\n1,1\n0,1\n")
    assert answer_columns(file) == answer_columns(file)


def test_a_file_that_is_not_csv_gets_a_clear_error():
    not_csv = io.BytesIO(bytes([0xFF, 0xFE, 0x00, 0x41]) * 50)
    with pytest.raises(ValueError, match="could not read this file"):
        answer_columns(not_csv)
    with pytest.raises(ValueError, match="could not read this file"):
        score_columns(not_csv)
