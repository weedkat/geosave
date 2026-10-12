import pytest

from geosave_engine.model.spec import ModelSpec, Ref


def _spec():
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"image": {"channels": 1}},
            "inputs": {"image": Ref("image")},
            "context": {"call": "builtins.dict", "kwargs": {"seen": Ref("image")}},
        }
    )


def test_context_recipe_round_trips(tmp_path):
    spec = _spec()

    assert ModelSpec.load(spec.save(tmp_path / "model_spec.yaml")) == spec


def test_context_reads_only_declared_rasters():
    with pytest.raises(ValueError, match="Context references .*row"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {"image": {"channels": 1}},
                "context": {"call": "builtins.dict", "kwargs": {"t": Ref("row.times")}},
            }
        )


def test_context_reads_only_what_the_inputs_read():
    with pytest.raises(ValueError, match="Context references .*optical"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {"optical": {"channels": 1}},
                "preprocessing": {
                    "image": {"call": "builtins.dict", "kwargs": {"d": Ref("optical")}}
                },
                "inputs": {"image": Ref("image")},
                "context": {"call": "builtins.dict", "kwargs": {"r": Ref("optical")}},
            }
        )
