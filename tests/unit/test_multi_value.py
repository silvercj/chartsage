import pandas as pd
from multi_value import detect_multi_value, explode_multi_value


def test_detects_comma_space_multi_value():
    s = pd.Series(["United States, India", "India, UK", "United States", "UK, Japan, India"] * 30)
    assert detect_multi_value(s) == ", "


def test_detects_pipe():
    s = pd.Series(["Drama|Comedy", "Comedy|Action", "Drama", "Action|Drama"] * 30)
    assert detect_multi_value(s) == "|"


def test_rejects_free_text_sentences():
    s = pd.Series([f"A unique, long descriptive sentence number {i} about a title." for i in range(200)])
    assert detect_multi_value(s) is None


def test_rejects_too_many_atoms():
    s = pd.Series([f"Actor{i}, Actor{i+1}, Actor{i+2}" for i in range(400)])
    assert detect_multi_value(s) is None


def test_rejects_single_value_categorical():
    s = pd.Series(["TV-MA", "PG-13", "R", "TV-14"] * 50)
    assert detect_multi_value(s) is None


def test_explode_counts_atoms():
    s = pd.Series(["A, B", "B, C", "A"])
    atoms = explode_multi_value(s, ", ")
    assert atoms.tolist() == ["A", "B", "B", "C", "A"]
    vc = atoms.value_counts()
    assert vc["A"] == 2 and vc["B"] == 2
