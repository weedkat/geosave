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
    return folder


def _run(
    name: str,
    workdir: Path,
    *,
    data: Path | None = None,
    without: str | None = None,
) -> nbformat.NotebookNode:
    """Execute one notebook in a working directory of its own.

    Args:
        name: Notebook file in `examples/`.
        workdir: Empty directory the notebook runs in.
        data: Downloaded data folder, linked as `data/`; the notebook's own
            download then only checks it. None copies `examples/data`.
        without: Folder of `examples/data` the notebook must write itself.

    Returns:
        The notebook with the outputs of this run.
    """
    if data is not None:
        (workdir / "data").symlink_to(data, target_is_directory=True)
    else:
        skipped = shutil.ignore_patterns(without, "dw_prepared") if without else None
        shutil.copytree(EXAMPLES / "data", workdir / "data", ignore=skipped)
    if (EXAMPLES / "configs").is_dir():
        shutil.copytree(EXAMPLES / "configs", workdir / "configs")
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": workdir}},
    )
    return client.execute()


@pytest.mark.integration
def test_imagery_runs(tmp_path: Path) -> None:
    _run("01_imagery.ipynb", tmp_path, without="dw_raw")

    assert (tmp_path / "data/dw_raw/label.parquet").is_file()


@pytest.mark.slow
def test_prepare_runs(tmp_path: Path) -> None:
    _run("02_prepare.ipynb", tmp_path)

    assert (tmp_path / "data/dw_prepared/image.parquet").is_file()


@pytest.mark.slow
@pytest.mark.integration
def test_basics_runs(tmp_path: Path, data: Path) -> None:
    notebook = _run("00_basics.ipynb", tmp_path, data=data)

    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) <= 8
    assert all(cell.outputs for cell in code)
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


@pytest.mark.slow
@pytest.mark.integration
def test_model_runs(tmp_path: Path, data: Path) -> None:
    notebook = _run("01_model.ipynb", tmp_path, data=data)

    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert list((tmp_path / "lightning_logs").glob("version_*/events.out.tfevents.*"))


@pytest.mark.slow
@pytest.mark.integration
def test_rasters_runs(tmp_path: Path, data: Path) -> None:
    notebook = _run("02_rasters.ipynb", tmp_path, data=data)

    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.outputs for cell in code)
    assert (tmp_path / "predicted.tif").is_file()


@pytest.mark.slow
@pytest.mark.integration
def test_bands_runs(tmp_path: Path, data: Path) -> None:
    notebook = _run("03_bands.ipynb", tmp_path, data=data)

    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.outputs for cell in code)
    assert list((tmp_path / "lightning_logs").glob("version_*/events.out.tfevents.*"))
