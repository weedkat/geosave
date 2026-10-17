from __future__ import annotations

from collections.abc import Mapping

import pytest

from geosave.model.spec import Ref, StageSpec


def stage(**calls: object) -> StageSpec:
    return StageSpec.model_validate(calls)


def test_stage_is_an_ordered_mapping_with_plain_dump_shape() -> None:
    spec = stage(
        first={"call": "builtins.dict"},
        second={"call": "builtins.list"},
    )

    assert isinstance(spec, Mapping)
    assert list(spec) == ["first", "second"]
    assert spec["first"].call == "builtins.dict"
    assert spec.model_dump() == {
        "first": {"call": "builtins.dict", "kwargs": {}},
        "second": {"call": "builtins.list", "kwargs": {}},
    }


def test_stage_validates_external_inputs_and_order() -> None:
    spec = stage(
        selected={"call": Ref("source.select")},
        result={"call": "builtins.dict", "kwargs": {"value": Ref("selected")}},
    )

    assert spec.external_inputs == ("source",)
    assert spec.validate_inputs({"source", "unused"}) == frozenset({"source"})


def test_stage_rejects_missing_and_forward_references() -> None:
    missing = stage(result={"call": Ref("source.method")})
    forward = stage(
        first={"call": Ref("second")},
        second={"call": "builtins.dict"},
    )

    with pytest.raises(ValueError, match="source"):
        missing.validate_inputs(set())
    with pytest.raises(ValueError, match="Forward.*second"):
        forward.validate_inputs(set())

    assert forward.validate_inputs({"second"}) == frozenset({"second"})


def test_stage_allows_explicit_rebinding() -> None:
    spec = stage(
        value={
            "call": Ref("scale"),
            "kwargs": {"value": Ref("value"), "factor": 3},
        }
    )

    assert spec.external_inputs == ("scale", "value")
    assert spec.validate_inputs({"scale", "value"}) == frozenset({"scale", "value"})


def test_empty_stage_has_no_inputs() -> None:
    spec = StageSpec.model_validate({})

    assert spec.external_inputs == ()
    assert spec.validate_inputs(set()) == frozenset()


_events: list[int] = []


def record(*, value: int) -> int:
    _events.append(value)
    return value


def test_stage_runs_declarations_in_order() -> None:
    _events.clear()
    declarations = stage(
        first={"call": f"{__name__}.record", "kwargs": {"value": 1}},
        second={"call": f"{__name__}.record", "kwargs": {"value": 2}},
    )

    assert declarations.run({}) == {"first": 1, "second": 2}
    assert _events == [1, 2]


def test_stage_returns_only_results_without_mutating_inputs() -> None:
    supplied = {"value": 3, "unused": object()}
    declarations = stage(
        squared={
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        },
        ratio={"call": Ref("squared.as_integer_ratio")},
    )

    assert declarations.run(supplied) == {"squared": 9, "ratio": (9, 1)}
    assert supplied["value"] == 3


def test_stage_validates_all_references_before_calling() -> None:
    _events.clear()
    declarations = stage(
        first={"call": f"{__name__}.record", "kwargs": {"value": 1}},
        second={"call": Ref("missing")},
    )

    with pytest.raises(ValueError, match="missing"):
        declarations.run({})

    assert _events == []


def test_stage_rejects_forward_references_before_calling() -> None:
    _events.clear()
    declarations = stage(
        first={"call": Ref("second")},
        second={"call": f"{__name__}.record", "kwargs": {"value": 2}},
    )

    with pytest.raises(ValueError, match="Forward.*second"):
        declarations.run({})

    assert _events == []


def test_stage_runs_explicit_rebinding() -> None:
    declarations = stage(
        value={
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        }
    )

    assert declarations.run({"value": 3}) == {"value": 9}
