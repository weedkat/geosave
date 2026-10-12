"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).parents[2] / "examples"
DATASET = "fatmur/dynamicworld-300"


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the example dataset once for every notebook that reads it."""
    from huggingface_hub import snapshot_download

    folder = tmp_path_factory.mktemp("data")
    snapshot_download(DATASET, repo_type="dataset", local_dir=folder / "dynamic_world")
    # Notebook 3 turns this polygon into a label.
    shutil.copy(EXAMPLES / "data" / "khdtk_wanadipa_batas.geojson", folder)
    return folder


def _run(name: str, workdir: Path, data: Path) -> list[nbformat.NotebookNode]:
    """Execute one notebook in a working directory of its own.

    Args:
        name: Notebook file in `examples/`.
        workdir: Empty directory the notebook runs in.
        data: Downloaded data folder, linked as `data/`; the notebook's own
            download then only checks it.

    Returns:
        The notebook's code cells with the outputs of this run.
    """
    (workdir / "data").symlink_to(data, target_is_directory=True)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": workdir}},
    )
    client.execute()
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _scalars(run: Path) -> dict[str, int]:
    """Count the points logged per TensorBoard scalar of one named run."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = EventAccumulator(str(run))
    events.Reload()
    return {tag: len(events.Scalars(tag)) for tag in events.Tags()["scalars"]}


@pytest.mark.slow
@pytest.mark.integration
def test_basics_runs(tmp_path: Path, data: Path) -> None:
    code = _run("00_basics.ipynb", tmp_path, data)

    assert len(code) <= 8
    assert all(cell.execution_count is not None for cell in code)
    source = "\n".join(cell.source for cell in code)
    # "stac" alone would match `gs.stack`, which this notebook does use.
    for word in (
        "open_vector",
        "cuts.",
        "StacClient",
        "chain_step",
        "ModelSpec",
        "crs",
    ):
        assert word not in source

    scalars = _scalars(tmp_path / "lightning_logs/basics")
    for tag in ("train_loss_step", "val_loss", "val_miou", "val_accuracy"):
        assert tag in scalars
    # One validation point per epoch: a single epoch would plot as a dot.
    assert scalars["val_miou"] > 1


@pytest.mark.slow
@pytest.mark.integration
def test_model_runs(tmp_path: Path, data: Path) -> None:
    code = _run("01_model.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(tmp_path / "lightning_logs/model")


@pytest.mark.slow
@pytest.mark.integration
def test_inputs_runs(tmp_path: Path, data: Path) -> None:
    code = _run("02_inputs.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(tmp_path / "lightning_logs/inputs")
    assert (tmp_path / "predicted.tif").is_file()


@pytest.mark.slow
@pytest.mark.integration
def test_sources_runs(tmp_path: Path, data: Path) -> None:
    code = _run("03_sources.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    # The project's catalog, its shareable copy, and the composite made from it.
    from geosave.geodata.stac import StacTable

    dataset = tmp_path / "data" / "my_dataset"
    table = StacTable(dataset / "catalog.parquet")
    assert table.collections() == {"label", "sentinel-2-l2a"}
    assert (dataset / "stac" / "catalog.json").is_file()
    for split in ("train", "val", "test"):
        assert list((dataset / split / "images").glob("*.tif"))
        assert list((dataset / split / "labels").glob("*.tif"))
    assert (tmp_path / "composite.tif").is_file()
