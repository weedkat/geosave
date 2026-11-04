from __future__ import annotations


import pytest
from pydantic import ValidationError

from geosave.model.spec import CallSpec, Ref


def mutate(*, items: list[int]) -> list[int]:
    items.append(2)
    return items


def test_call_invokes_imports_bound_methods_and_nested_references() -> None:
    imported = CallSpec.model_validate(
        {
            "call": "builtins.dict",
            "kwargs": {
                "nested": [Ref("source")],
                "literal": {"ref": "source"},
                "text": "!ref source",
            },
        }
    )
    source = object()

    assert imported.invoke({"source": source}) == {
        "nested": [source],
        "literal": {"ref": "source"},
        "text": "!ref source",
    }
    assert CallSpec(call=Ref("number.as_integer_ratio")).invoke({"number": 1.5}) == (
        3,
        2,
    )


def test_call_rebuilds_literal_containers() -> None:
    spec = CallSpec(call=Ref("mutate"), kwargs={"items": [1]})

    first = spec.invoke({"mutate": mutate})
    second = spec.invoke({"mutate": mutate})

    assert first == second == [1, 2]
    assert first is not second


def test_call_binds_none_as_the_complete_result() -> None:
    items: dict[str, int] = {}
    spec = CallSpec(call=Ref("items.update"), kwargs={"answer": 42})

    assert spec.invoke({"items": items}) is None
    assert items == {"answer": 42}


def test_reference_reads_a_key_no_attribute_can_name() -> None:
    reference = Ref('scene["view:sun_azimuth"].real')

    assert reference.root == "scene"
    assert reference.resolve({"scene": {"view:sun_azimuth": 1.5}}) == 1.5


def test_call_rejects_cyclic_keyword_values() -> None:
    kwargs: dict[str, object] = {}
    kwargs["self"] = kwargs

    with pytest.raises((ValueError, ValidationError), match="Cyclic|recursion"):
        CallSpec.model_validate({"call": "builtins.dict", "kwargs": kwargs})


def test_call_error_note_identifies_the_declared_target() -> None:
    def fail() -> None:
        raise RuntimeError("failed")

    spec = CallSpec(call=Ref("fail"))

    with pytest.raises(RuntimeError) as caught:
        spec.invoke({"fail": fail})

    assert caught.value.__notes__ == ["While invoking Ref('fail')"]
