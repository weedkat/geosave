from concurrent.futures import ThreadPoolExecutor
from graphlib import CycleError
from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from tests.model.spec import operations
from geosave.model.spec import Input, ModelSpec, Step
from geosave.model.spec.reference import Reference


TEXT = Path(__file__).with_name("model_spec.yaml").read_text()
POLYGON = {
    "type": "Polygon",
    "coordinates": [[[107, -7], [108, -7], [108, -6], [107, -7]]],
}


@pytest.fixture
def spec():
    return ModelSpec.from_yaml(TEXT)


def test_load_and_plan_are_inert(monkeypatch):
    from geosave.model.spec import model as model_spec

    def reject(name):
        raise AssertionError("Imported a callable during loading or planning")

    monkeypatch.setattr(model_spec, "import_module", reject)
    spec = ModelSpec.from_yaml(TEXT)
    plan = spec.plan(entry="from_area", output="classes")
    assert plan["inputs"] == ("geometry", "time", "batch_size")
    assert plan["steps"] == ("from_area", "checked", "prepared", "scores", "classes")


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("from_raster", {"raster": operations.scene()}),
        ("from_area", {"geometry": POLYGON, "time": "2025-06"}),
        (
            "from_center",
            {
                "longitude": 107.0,
                "latitude": -7.0,
                "width_m": 40.0,
                "height_m": 40.0,
                "time": "2025-06",
            },
        ),
    ],
)
def test_sources_converge_to_same_contract(spec, name, arguments):
    result = spec.run(entry=name, output="classes", **arguments)
    np.testing.assert_array_equal(result.label.values, np.zeros((4, 4), dtype="uint8"))
    assert result.gs.geobox == operations.scene().gs.geobox


def test_intermediate_preserves_laziness(spec):
    events = []
    with Callback(posttask=lambda *args: events.append(1)):
        prepared = spec.run(
            entry="from_raster", output="prepared", raster=operations.scene()
        )
    assert not events
    assert isinstance(prepared.red.data, da.Array)
    np.testing.assert_array_equal(prepared.red.values, np.ones((4, 4)))


def test_output_path_needed_only_for_writer(spec, tmp_path):
    raster = operations.scene()
    result = spec.run(entry="from_raster", output="classes", raster=raster)
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "classes.tif"
    saved = spec.run(
        entry="from_raster", output="saved", raster=raster, destination=str(path)
    )
    reopened = operations.open_raster(saved)
    np.testing.assert_array_equal(reopened.label.values, result.label.values)
    assert reopened.gs.geobox == raster.gs.geobox
    assert reopened.label.dtype == np.dtype("uint8")


def test_shared_dependency_executes_once(spec, monkeypatch):
    original = operations.prepare
    calls = []

    def prepare(raster):
        calls.append(1)
        return original(raster)

    monkeypatch.setattr(operations, "prepare", prepare)
    result = spec.run(
        entry="from_raster", output=["prepared", "classes"], raster=operations.scene()
    )
    assert len(calls) == 1
    assert set(result) == {"prepared", "classes"}


def test_concurrent_invocations_do_not_share_results(spec):
    def predict(value):
        result = spec.run(
            entry="from_raster", output="classes", raster=operations.scene(value)
        )
        return int(result.label.values[0, 0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(predict, [2.0, -2.0])) == [0, 1]


def test_downstream_option_does_not_enter_acquisition(spec, monkeypatch):
    calls = []

    def predict(raster, batch_size):
        calls.append(batch_size)
        return xr.DataArray(
            np.zeros((2, 4, 4)), dims=("class", "y", "x")
        ).gs.write_grid(raster.gs.geobox)

    monkeypatch.setattr(operations, "predict", predict)
    spec.run(
        entry="from_area",
        output="classes",
        geometry=POLYGON,
        time="2025-06",
        batch_size=7,
    )
    assert calls == [7]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"raster": operations.scene(), "unexpected": 1},
    ],
)
def test_missing_or_unused_inputs_fail_before_calling(spec, monkeypatch, arguments):
    def reject(raster):
        raise AssertionError("Entry executed before argument preflight")

    monkeypatch.setattr(operations, "accept", reject)
    with pytest.raises(ValueError):
        spec.run(entry="from_raster", output="classes", **arguments)


def test_missing_output_fails_before_preparation(spec, monkeypatch):
    def reject(raster):
        raise AssertionError("Entry executed before missing destination check")

    monkeypatch.setattr(operations, "accept", reject)
    with pytest.raises(ValueError, match="input 'destination' is missing"):
        spec.run(entry="from_raster", output="saved", raster=operations.scene())


def test_invalid_raster_prevents_model_execution(spec, monkeypatch):
    def reject(raster, batch_size):
        raise AssertionError("Model executed before raster validation")

    monkeypatch.setattr(operations, "predict", reject)
    with pytest.raises(ValueError, match="Expected one red"):
        spec.run(entry="from_raster", output="classes", raster=xr.Dataset())


def test_references_select_names_and_attributes():
    values = {"raster": operations.scene()}
    assert Reference("raster").read_value(values) is values["raster"]
    assert (
        Reference("raster.gs.geobox").read_value(values) == values["raster"].gs.geobox
    )


@pytest.mark.parametrize(
    "path",
    ["raster.compute()", "x + 1", "x[y]", "x[0:2]", 'context["grid"]', "outputs[0]"],
)
def test_references_do_not_evaluate_expressions(path):
    with pytest.raises(ValueError):
        Reference(path)


def test_bound_method_call(tmp_path):
    spec = ModelSpec.from_yaml("""schema_version: 1
inputs:
  value: {}
steps:
  initial:
    call: tests.model.spec.operations.accept
    kwargs: {raster: !ref value}
  answer:
    call: !ref entry.upper
""")
    assert spec.run(entry="initial", output="answer", value="hello") == "HELLO"
    restored = ModelSpec.load(spec.save(tmp_path / "recipe.yaml"))
    assert restored.run(entry="initial", output="answer", value="hello") == "HELLO"


@pytest.mark.parametrize(
    "call", ["tests.model.spec.operations.accept", Reference("entry.upper")]
)
def test_call_argument_errors_report_step_after_dependencies_run(monkeypatch, call):
    calls = []

    def accept(raster):
        calls.append(raster)
        return raster

    monkeypatch.setattr(operations, "accept", accept)
    spec = ModelSpec(
        inputs={"text": Input()},
        steps={
            "accepted": Step(
                call="tests.model.spec.operations.accept",
                kwargs={"raster": Reference("text")},
            ),
            "failed": Step(call=call, kwargs={"unexpected": Reference("entry")}),
        },
        entry="accepted",
        output="failed",
    )
    with pytest.raises(TypeError) as error:
        spec.run(text="hello")
    assert calls == ["hello"]
    notes = "\n".join(error.value.__notes__)
    assert "Step 'failed'" in notes
    assert "output 'failed'" in notes


def test_selector_names_cannot_be_inputs():
    with pytest.raises(ValueError, match="reserved"):
        ModelSpec.from_yaml(TEXT.replace("  raster: {}", "  output: {}"))


def test_input_mutation_is_not_prevented(spec, monkeypatch):
    def mutate(raster):
        raster.attrs["changed"] = True
        return raster

    monkeypatch.setattr(operations, "prepare", mutate)
    raster = operations.scene()
    spec.run(entry="from_raster", output="prepared", raster=raster)
    assert raster.attrs["changed"] is True


def test_unknown_reference_and_duplicate_names_fail():
    with pytest.raises(ValueError, match="unknown reference"):
        ModelSpec.from_yaml(TEXT.replace("raster: !ref checked", "raster: !ref absent"))
    with pytest.raises(ValueError, match="unique strings"):
        ModelSpec.from_yaml(TEXT + "\ninputs: {}\n")
    with pytest.raises(ValueError, match="distinct"):
        ModelSpec.from_yaml(TEXT.replace("  raster: {}", "  prepared: {}"))


def test_cycles_include_selected_source():
    with pytest.raises(CycleError):
        ModelSpec.from_yaml(
            TEXT.replace("raster: !ref checked", "raster: !ref classes")
        ).plan(entry="from_raster", output="classes")
    with pytest.raises(CycleError):
        ModelSpec.from_yaml(
            TEXT.replace(
                "kwargs: {raster: !ref raster}", "kwargs: {raster: !ref entry}"
            )
        ).plan(entry="from_raster", output="classes")


def test_missing_selection_without_defaults_and_no_intermediate_override(spec):
    without_defaults = ModelSpec.from_yaml(
        TEXT.replace("entry: from_raster\n", "").replace("output: classes\n", "")
    )
    with pytest.raises(ValueError, match="No workflow entry"):
        without_defaults.run(output="classes", raster=operations.scene())
    with pytest.raises(ValueError, match="No workflow output"):
        without_defaults.run(entry="from_raster", raster=operations.scene())
    with pytest.raises(ValueError, match="unused"):
        spec.run(
            entry="from_raster",
            output="classes",
            raster=operations.scene(),
            prepared=operations.scene(),
        )


def test_file_source_roundtrip(spec, tmp_path):
    path = operations.save(operations.scene(-2.0), str(tmp_path / "source.tif"))
    result = spec.run(entry="from_path", output="classes", path=path)
    np.testing.assert_array_equal(result.label.values, np.ones((4, 4), dtype="uint8"))


@pytest.mark.parametrize("path", ["recipe.yaml", "memory://modelspec/recipe.yaml"])
def test_recipe_roundtrip_preserves_references_defaults_and_execution(tmp_path, path):
    target = path if path.startswith("memory:") else tmp_path / path
    recipe = ModelSpec.from_yaml(TEXT)
    saved = recipe.save(target)
    restored = ModelSpec.load(saved)
    expected = recipe.run(raster=operations.scene())
    actual = restored.run(raster=operations.scene())
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "schema_version: true",
        "schema_version: 3",
        "schema_version: 1\ninputs: []",
        "schema_version: 1\ninputs: {a: null}",
    ],
)
def test_malformed_documents_fail_at_load(text):
    with pytest.raises(ValueError):
        ModelSpec.from_yaml(text)


def test_mutable_defaults_are_fresh_and_nested_references_resolve():
    spec = ModelSpec.from_yaml("""schema_version: 1
inputs:
  value: {default: []}
steps:
  initial:
    call: builtins.dict
    kwargs: {items: [!ref value]}
""")
    first = spec.run(entry="initial", output="initial")
    first["items"][0].append(1)
    assert spec.run(entry="initial", output="initial") == {"items": [[]]}


def test_planning_does_not_resolve_properties_and_execution_retains_identity():
    class Object:
        @property
        def value(self):
            raise RuntimeError("property executed")

    spec = ModelSpec.from_yaml("""schema_version: 1
inputs: {object: {}}
steps:
  answer:
    call: builtins.dict
    kwargs: {value: !ref object.value}
""")
    spec.plan(entry="answer", output="answer")
    with pytest.raises(RuntimeError, match="property executed") as error:
        spec.run(entry="answer", output="answer", object=Object())
    assert "answer" in "\n".join(error.value.__notes__)


def test_builtin_raster_validation_preserves_native_lazy_values():
    from geosave.model.processing import validate_raster

    raster = operations.scene()
    with Callback(pretask=lambda *args: pytest.fail("Read pixels during validation")):
        assert validate_raster(raster) is raster
    with pytest.raises(ValueError):
        validate_raster(xr.Dataset({"red": (("y", "x"), np.ones((2, 2)))}))


def test_valid_input_self_does_not_collide_with_python_receiver():
    spec = ModelSpec.from_yaml("""schema_version: 1
inputs: {self: {}}
steps:
  answer:
    call: builtins.dict
    kwargs: {value: !ref self}
""")
    assert spec.run(entry="answer", output="answer", **{"self": 3}) == {"value": 3}


def test_documented_recipe_runs_through_save(tmp_path):
    recipe = ModelSpec.load(Path(__file__).parents[3] / "docs/guides/model_spec.yaml")
    raster = operations.scene()
    result = recipe.run(entry="checked", output="encoded", raster=raster)
    np.testing.assert_array_equal(result.numpy(), raster.to_array().values)
    path = recipe.run(
        entry="checked",
        output="saved",
        raster=raster,
        destination=str(tmp_path / "scene.tif"),
    )
    reopened = operations.open_raster(path)
    np.testing.assert_array_equal(reopened.red.values, raster.red.values)
    assert reopened.gs.geobox == raster.gs.geobox


def test_defaults_and_overrides_select_real_paths(spec, tmp_path):
    raster = operations.scene()
    expected = spec.run(entry="from_raster", output="classes", raster=raster)
    xr.testing.assert_identical(spec.run(raster=raster), expected)

    source = operations.save(raster, str(tmp_path / "input.tif"))
    actual = spec.run(entry="from_path", path=source)
    np.testing.assert_array_equal(actual.label.values, expected.label.values)
    assert actual.gs.geobox == expected.gs.geobox
    outputs = spec.run(output=["prepared", "classes"], raster=raster)
    xr.testing.assert_identical(outputs["classes"], expected)
    assert isinstance(outputs["prepared"].red.data, da.Array)


def test_selected_entry_must_participate_before_operations_are_imported(
    spec, monkeypatch
):
    from geosave.model.spec import model

    def reject(name):
        pytest.fail("Imported code before rejecting a disconnected entry")

    monkeypatch.setattr(model, "import_module", reject)
    with pytest.raises(ValueError, match="does not use entry"):
        spec.run(entry="from_path", output="from_raster", raster=operations.scene())


def test_constructor_and_yaml_share_workflow_semantics(tmp_path):
    spec = ModelSpec(
        inputs={"value": Input()},
        steps={
            "accepted": Step(
                call="builtins.dict", kwargs={"value": Reference("value")}
            ),
            "returned": Step(
                call="builtins.dict", kwargs={"wrapped": Reference("entry")}
            ),
        },
        entry="accepted",
        output=["accepted", "returned"],
    )
    value = object()
    result = spec.run(value=value)
    assert result["accepted"]["value"] is value
    assert result["returned"]["wrapped"] is result["accepted"]

    restored = ModelSpec.load(spec.save(tmp_path / "recipe.yaml"))
    assert restored.run(value=value)["returned"]["wrapped"]["value"] is value


def test_import_failure_identifies_output_step_and_configured_operation():
    spec = ModelSpec(
        steps={"prepared": Step(call="missing_modelspec_package.prepare")},
        entry="prepared",
        output="prepared",
    )
    with pytest.raises(ModuleNotFoundError) as error:
        spec.run()
    notes = "\n".join(error.value.__notes__)
    assert "prepared" in notes
    assert "missing_modelspec_package.prepare" in notes
    assert "installed" in notes


def test_failed_reference_identifies_argument_step_and_value_type():
    spec = ModelSpec(
        inputs={"raster": Input()},
        steps={
            "decoded": Step(
                call="builtins.dict",
                kwargs={"grid": {"nested": Reference("raster.absent_grid")}},
            )
        },
        entry="decoded",
        output="decoded",
    )
    with pytest.raises(AttributeError) as error:
        spec.run(raster=xr.Dataset())
    notes = "\n".join(error.value.__notes__)
    for text in ("raster.absent_grid", "Dataset", "argument 'grid'", "Step 'decoded'"):
        assert text in notes


def test_null_default_remains_distinct_from_required_input_after_saving(tmp_path):
    spec = ModelSpec.from_yaml("""schema_version: 1
entry: collected
output: collected
inputs:
  required: {}
  optional: {default: null}
steps:
  collected:
    call: builtins.dict
    kwargs: {required: !ref required, optional: !ref optional}
""")
    restored = ModelSpec.load(spec.save(tmp_path / "recipe.yaml"))
    assert restored.run(required=3) == {"required": 3, "optional": None}
    with pytest.raises(ValueError, match="input 'required' is missing"):
        restored.run()


def test_nested_steps_can_use_document_field_names(tmp_path):
    spec = ModelSpec.from_yaml("""schema_version: 1
entry: inputs
output: output
steps:
  inputs:
    call: builtins.dict
    kwargs: {answer: 42}
  output:
    call: builtins.dict
    kwargs: {value: !ref entry}
""")
    restored = ModelSpec.load(spec.save(tmp_path / "recipe.yaml"))
    assert restored.run() == {"value": {"answer": 42}}
