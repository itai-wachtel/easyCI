"""Checks that pasted text and CSV files are read correctly."""

import io

import pytest

from parsing import parse_scores, scores_from_csv


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


def test_csv_with_two_columns():
    file = io.StringIO("baseline,ours\n83.1,84.2\n83.9,83.8\n")
    assert scores_from_csv(file) == ("baseline", [83.1, 83.9], "ours", [84.2, 83.8])


def test_csv_columns_can_differ_in_length():
    file = io.StringIO("baseline,ours\n83.1,84.2\n83.9,\n")
    name_a, a, name_b, b = scores_from_csv(file)
    assert a == [83.1, 83.9]
    assert b == [84.2]


def test_csv_skips_text_columns():
    file = io.StringIO("seed_name,baseline,ours\nfirst,83.1,84.2\nsecond,83.9,83.8\n")
    assert scores_from_csv(file)[0] == "baseline"


def test_csv_with_one_column_is_rejected():
    with pytest.raises(ValueError, match="two columns"):
        scores_from_csv(io.StringIO("baseline\n83.1\n83.9\n"))
