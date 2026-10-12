import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata import open_vector, raster, stac, stack
from tests.geodata.conftest import stack_templates

GROUPS = ("s2", "s1", "dem", "label")


@pytest.fixture
def grid() -> GeoBox:
    # 50 rows by 70 columns
    return GeoBox.from_bbox(
        (300000, 5000000, 300700, 5000500), "EPSG:32633", resolution=10
    )


def _series(name: str, dates: list[str], grid: GeoBox):
    times = pd.to_datetime(dates)
    rng = np.random.default_rng(len(dates))
    values = rng.integers(1, 9000, (len(times), *grid.shape)).astype("uint16")
    return raster({name: (("time", "y", "x"), values)}, grid, time=times)


@pytest.fixture
def sample(grid):
    """Optical and radar on their own dates, a timeless DEM, and a label."""
    rng = np.random.default_rng(0)
    return stack(
        {
            "s2": _series(
                "red",
                ["2024-01-10", "2024-03-08", "2024-03-28", "2024-04-12", "2024-07-03"],
                grid,
            ),
            "s1": _series("vv", ["2024-03-05", "2024-04-04", "2024-07-01"], grid),
            "dem": raster(
                {"height": (("y", "x"), rng.random(grid.shape).astype("float32"))}, grid
            ),
            "label": raster(
                {"label": (("y", "x"), rng.integers(0, 3, grid.shape).astype("uint8"))},
                grid,
            ),
        }
    )


@pytest.fixture
def items(sample, tmp_path):
    """The sample written as COGs and catalogued, read back from its table."""
    files = sample.gs.to_cog(stack_templates(sample, tmp_path / "s0"))
    rows = stac.create_items(files, id="s0")
    return open_vector(rows.gs.to_geoparquet(tmp_path / "items.parquet"))
