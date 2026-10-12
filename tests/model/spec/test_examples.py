from pathlib import Path
import subprocess
import sys

from geosave.cli.core.workspace import create_workspace
from geosave.model.spec import ModelSpec, Ref


def test_shipped_model_spec_round_trips_as_inert_declarations(tmp_path):
    path = (
        Path(__file__).parents[3]
        / "src/geosave/templates/workspaces/segmentation/configs/model_spec.yaml"
    )

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path / "model_spec.yaml"))

    assert restored == spec
    assert spec.rasters["sentinel_2_l2a"].variables == (
        "B02",
        "B03",
        "B04",
        "B08",
    )
    assert spec.preprocessing["image"].call == Ref("sentinel_2_l2a.gs.mask_and_scale")
    assert spec.preprocessing["image"].kwargs == {}


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    create_workspace(tmp_path, "segmentation")

    result = subprocess.run(
        [sys.executable, "scripts/prepare_example.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "B08" in result.stdout
    assert "0.6" in result.stdout
