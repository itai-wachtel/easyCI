"""Checks that pasted text and CSV files are read correctly."""

import io

import pytest

from parsing import answers_from_csv, parse_scores, scores_from_csv


def test_answers_csv_with_ones_and_zeros():
    file = io.StringIO("baseline,ours\n1,1\n0,1\n1,0\n")
    assert answers_from_csv(file) == (
        "baseline", [True, False, True], "ours", [True, True, False],
    )


def test_answers_csv_with_true_false_words():
    file = io.StringIO("baseline,ours\nTrue,false\nFALSE,true\n")
    assert answers_from_csv(file) == ("baseline", [True, False], "ours", [False, True])


def test_answers_csv_skips_other_columns():
    file = io.StringIO("example_id,text,baseline,ours\n17,hello,1,0\n18,bye,0,0\n")
    name_a, a, name_b, b = answers_from_csv(file)
    assert (name_a, name_b) == ("baseline", "ours")


def test_answers_csv_with_a_missing_value_is_rejected():
    with pytest.raises(ValueError, match="two columns of 1"):
        answers_from_csv(io.StringIO("baseline,ours\n1,1\n0,\n"))


def test_answers_csv_with_scores_instead_of_answers_is_rejected():
    with pytest.raises(ValueError, match="two columns of 1"):
        answers_from_csv(io.StringIO("baseline,ours\n83.1,84.2\n83.9,83.8\n"))


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
