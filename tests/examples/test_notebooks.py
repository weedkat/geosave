"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).parents[2] / "examples"
DATASET = "fatmur/dynamicworld-300"
EXERCISE = "# Your code here"
SOLUTION = re.compile(r"<summary>Solution</summary>\s*```python\n(.*?)```", re.DOTALL)


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the example dataset once for every notebook that reads it."""
    from huggingface_hub import snapshot_download

    folder = tmp_path_factory.mktemp("data")
    snapshot_download(DATASET, repo_type="dataset", local_dir=folder / "dynamic_world")
    return folder


def fill_solutions(notebook: nbformat.NotebookNode) -> None:
    """Put each exercise's folded solution into the empty cell below it.

    Solutions live in markdown, so without this they would never run and
    would drift from the API. Exercises with only a hint keep their empty cell.
    """
    previous = None
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip() == EXERCISE:
            if previous is not None:
                match = SOLUTION.search(previous.source)
                if match is not None:
                    cell.source = match.group(1)
        if cell.cell_type == "markdown":
            previous = cell


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
    fill_solutions(notebook)
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
    # Notebook 3 uses both the paired tiles and inference imagery from notebook 2.
    if not all(
        (data / folder / "catalog.parquet").is_file()
        for folder in ("khdtk", "my_dataset")
    ):
        (tmp_path / "sources").mkdir()
        test_sources_runs(tmp_path / "sources", data)

    workdir = tmp_path / "inputs"
    workdir.mkdir()
    code = _run("03_inputs.ipynb", workdir, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(workdir / "lightning_logs/03_train/version_0")
    # The forest's predicted land cover, one map per month of the season.
    import geosave as gs

    cover = gs.open_vector(data / "khdtk_pred" / "catalog.parquet")
    assert set(cover["collection"]) == {"land_cover"} and len(cover) == 6


def test_catalog_examples_load_chips_and_restore_pixels(tmp_path):
    """Execute the lessons' data functions with small real local STAC assets."""
    import numpy as np
    import torch
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from torch.utils.data import Dataset

    import geosave as gs
    from geosave.geodata import io
    from tests.conftest import build_dataset
    from tests.geodata.conftest import create_test_collection

    grid = GeoBox.from_bbox(
        (500000, 9200000, 500080, 9200080), crs="EPSG:32749", resolution=10
    )
    bands = ["B02", "B03", "B04", "B08"]
    pixels = {}
    for band in bands:
        pixels[band] = (("y", "x"), np.full((8, 8), 2000, dtype="uint16"))
    image = build_dataset(pixels, grid, nodata=0).assign_coords(
        time=np.datetime64("2025-01-15")
    )
    collection = create_test_collection(
        "sentinel-2-l2a", description="Local test imagery"
    )
    source = image.gs.to_catalog(tmp_path / "source", collection=collection)
    path = source.gs.to_geoparquet(tmp_path / "source.parquet")
    client = gs.StacClient.open(path)
    namespace = {
        "gs": gs,
        "io": io,
        "xr": xr,
        "client": client,
        "collection": collection,
        "query": gs.StacQuery(collections=[collection.id], max_items=1),
        "options": dict(bands=bands, dtype="uint16", nodata=0),
    }
    sources = nbformat.read(EXAMPLES / "02_sources.ipynb", as_version=4)
    cell = next(cell for cell in sources.cells if "def fetch_image(" in cell.source)
    exec(compile(cell.source, "02_sources.ipynb", "exec"), namespace)
    label = build_dataset(
        {"label": (("y", "x"), np.ones((8, 8), dtype="uint8"))}, grid, nodata=255
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.formats.geotiff.write_cog(label, tmp_path / "label.tif")
    table = namespace["save_tile"](label_path, tmp_path / "paired")
    sample = table.gs.sample(pair=["sentinel-2-l2a", "label"], tolerance="1D")[0]
    tree = sample.gs.to_datatree()
    assert tree.gs.geobox == grid

    inputs = nbformat.read(EXAMPLES / "03_inputs.ipynb", as_version=4)
    cell = next(cell for cell in inputs.cells if "def read_image(" in cell.source)
    namespace.update(Dataset=Dataset, IGNORE_INDEX=255)
    exec(compile(cell.source, "03_inputs.ipynb", "exec"), namespace)
    dataset = namespace["DynamicWorld"]([tree], chip_size=4, overlap=2)
    image_tensor, label_tensor = dataset[0]
    assert image_tensor.shape == (4, 4, 4)
    assert label_tensor.shape == (4, 4)
    assert label_tensor.dtype == torch.int64
    outputs = np.ones((len(dataset), 2, 4, 4), dtype="float32")
    (merged,) = dataset.merge(outputs)
    assert merged.gs.geobox == grid
    np.testing.assert_allclose(merged.values, 1)
    imagery = xr.DataTree.from_dict({"sentinel-2-l2a": tree["sentinel-2-l2a"]})
    inference = namespace["DynamicWorld"]([imagery], chip_size=4)
    assert inference[0].shape == (4, 4, 4)
