import pytest

from geosave.model.spec import ModelSpec, Ref, StageSpec

_events: list[int] = []


def record(*, value: int) -> int:
    _events.append(value)
    return value


def stage(**calls: object) -> StageSpec:
    return StageSpec.model_validate(calls)


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


def test_stage_allows_explicit_rebinding() -> None:
    declarations = stage(
        value={
            "call": "builtins.pow",
            "kwargs": {"base": Ref("value"), "exp": 2},
        }
    )

    assert declarations.run({"value": 3}) == {"value": 9}


def test_model_spec_rejects_undeclared_preprocessing_roots() -> None:
    with pytest.raises(ValueError, match="declared rasters.*scale.*value"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {},
                "preprocessing": {
                    "scaled": {
                        "call": Ref("scale"),
                        "kwargs": {"value": Ref("value")},
                    }
                },
            }
        )
