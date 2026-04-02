from src.lib.utils import tokenise_string
from unittest.mock import patch


def test_tokenise_lowercases_and_stems():
    result = tokenise_string("Running QUICKLY")
    assert all(t == t.lower() for t in result)
    assert "run" in result
    assert "quickli" in result


def test_tokenise_strips_asterisks_and_short_tokens():
    result = tokenise_string("pain***killer is a x good remedy")
    assert all(len(t) > 1 for t in result)
    assert "x" not in result
    assert "*" not in "".join(result)


def test_tokenise_removes_stopwords():
    result = tokenise_string("the quick brown fox")
    assert "the" not in result
    assert "a" not in result