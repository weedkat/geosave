"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).parents[2] / "examples"
DATASET = "fatmur/dynamicworld-300"
EXERCISE = "# Your code here"


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the example dataset once for every notebook that reads it."""
    from huggingface_hub import snapshot_download

    folder = tmp_path_factory.mktemp("data")
    snapshot_download(DATASET, repo_type="dataset", local_dir=folder / "dynamic_world")
    return folder


def _run(name: str, workdir: Path, data: Path) -> list[nbformat.NotebookNode]:
    """Execute one notebook in a working directory of its own.

    Args:
        name: Notebook file in `examples/`.
        workdir: Empty directory the notebook runs in.
        data: Downloaded data folder, linked as `data/`; the notebook's own
            download then only checks it.

    Returns:
        The notebook's code cells with the outputs of this run, without the
        empty cells its exercises leave for the reader.
    """
    (workdir / "data").symlink_to(data, target_is_directory=True)
    # Notebooks 2 and 3 read the forest outline shipped beside them.
    shutil.copy(EXAMPLES / "khdtk.geojson", workdir)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": workdir}},
    )
    client.execute()
    return [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip() != EXERCISE
    ]


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

    assert all(cell.execution_count is not None for cell in code)
    source = "\n".join(cell.source for cell in code)
    # "stac" alone would match `gs.stack`, which this notebook does use.
    for word in (
        "open_vector",
        ".gs.stacks(",
        "StacClient",
        "chain_step",
        "ModelSpec",
        "crs",
    ):
        assert word not in source

    scalars = _scalars(tmp_path / "lightning_logs/00_basics/version_0")
    for tag in ("train_loss_step", "val_loss", "val_miou", "val_accuracy"):
        assert tag in scalars
    # One validation point per epoch: a single epoch would plot as a dot.
    assert scalars["val_miou"] > 1


@pytest.mark.slow
@pytest.mark.integration
def test_model_runs(tmp_path: Path, data: Path) -> None:
    code = _run("01_model.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(tmp_path / "lightning_logs/01_model/version_0")


@pytest.mark.slow
@pytest.mark.integration
def test_sources_runs(tmp_path: Path, data: Path) -> None:
    code = _run("02_sources.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    # The project's catalog and the pairs it lists.
    import geosave as gs

    dataset = tmp_path / "data" / "my_dataset"
    table = gs.StacClient.open(dataset / "catalog.parquet")
    assert table.collections() == {"label", "sentinel-2-l2a"}
    # Each split keeps image and label filenames paired in its own folders.
    for split, count in [("train", 6), ("val", 2), ("test", 2)]:
        labels = sorted(path.name for path in (dataset / split / "label").glob("*.tif"))
        images = sorted(
            path.name for path in (dataset / split / "sentinel-2-l2a").glob("*.tif")
        )
        assert len(labels) == count and images == labels
    splits = gs.open_vector(dataset / "catalog.parquet")["split"].value_counts()
    assert splits.to_dict() == {"train": 12, "val": 4, "test": 4}
    # The forest season notebook 3 predicts over, one picture per month.
    forest = gs.StacClient.open(tmp_path / "data" / "khdtk" / "catalog.parquet")
    assert forest.collections() == {"sentinel-2-l2a"}


@pytest.mark.slow
@pytest.mark.integration
def test_inputs_runs(tmp_path: Path, data: Path) -> None:
    # Notebook 3 uses the original training data and notebook 2's inference imagery.
    if not (data / "khdtk" / "catalog.parquet").is_file():
        (tmp_path / "sources").mkdir()
        test_sources_runs(tmp_path / "sources", data)

    workdir = tmp_path / "inputs"
    workdir.mkdir()
    code = _run("03_inputs.ipynb", workdir, data)

    assert all(cell.execution_count is not None for cell in code)
    source = "\n".join(cell.source for cell in code)
    # Specs belong to later notebooks; cuts are reached through the accessor.
    for word in ("ModelSpec", "gs.cuts"):
        assert word not in source
    assert "val_miou" in _scalars(workdir / "lightning_logs/03_train/version_0")
    # The forest's predicted land cover, one map per month of the season.
    import geosave as gs

    cover = gs.open_vector(data / "khdtk_pred" / "catalog.parquet")
    assert set(cover["collection"]) == {"land_cover"} and len(cover) == 6
