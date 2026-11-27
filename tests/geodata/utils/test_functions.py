import pytest

from geosave.geodata.utils.functions import resolve_fn


def double(value: int) -> int:
    return value * 2


def test_a_name_selects_its_function_and_a_function_passes_as_given():
    functions = {"double": double}

    assert resolve_fn("double", functions) is double
    assert resolve_fn(abs, functions) is abs


def test_an_unknown_name_lists_the_known_names():
    with pytest.raises(KeyError, match=r"'triple' is not one of \['double'\]"):
        resolve_fn("triple", {"double": double})
