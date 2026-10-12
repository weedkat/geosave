import pytest
from prefect import Flow

from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.flows import prepare_dense_data


def test_layers_export_only_supported_operations() -> None:
    assert flows.__all__ == ["ingest", "prepare_dense_data"]
    assert tasks.__all__ == []
    assert isinstance(flows.ingest, Flow)
    assert isinstance(flows.prepare_dense_data, Flow)
    assert prepare_dense_data is flows.prepare_dense_data
    for helper in (
        "SampleFormat",
        "find_labels",
        "open_sample",
        "read_sample_metadata",
        "sample_assets",
        "write_manifest",
        "write_sample",
    ):
        assert not hasattr(tasks, helper)


def test_dense_flow_is_explicitly_pending_without_side_effects(tmp_path):
    with pytest.raises(NotImplementedError, match="Dense sample catalog"):
        prepare_dense_data.fn(
            str(tmp_path / "labels"),
            output=str(tmp_path / "prepared"),
            spec=str(tmp_path / "model_spec.yaml"),
        )
    assert list(tmp_path.iterdir()) == []
