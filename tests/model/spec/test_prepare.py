"""Prepare: a raw item table becomes the catalog of what the model reads."""

from __future__ import annotations

from datetime import UTC, datetime

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata import GeoStack, GeoVector, cuts, open_vector, raster, stack
from geosave.geodata.stac import create_items
from geosave.model.spec import ModelSpec, Ref

WHEN = datetime(2025, 6, 1, tzinfo=UTC)


def doubled(*, data: xr.Dataset) -> xr.Dataset:
    return data * 2


def _spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["nir"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.doubled",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "inputs": {"image": Ref("image")},
        }
    )


def _optical(index: int) -> xr.Dataset:
    left = 500_000 + index * 1000
    grid = GeoBox.from_bbox(
        (left, 9_000_000, left + 80, 9_000_000 + 80), "EPSG:32748", resolution=10
    )
    plane = np.arange(64, dtype="float32").reshape(8, 8) + 100 * index
    return raster({"red": (("y", "x"), plane), "nir": (("y", "x"), plane + 0.5)}, grid)


def _raw(root) -> GeoVector:
    """Write two samples, optical beside a label, and return their item table."""
    tables = []
    for index in range(2):
        optical = _optical(index)
        label = raster(
            {"label": (("y", "x"), np.full((8, 8), index, "uint8"))},
            optical.gs.geobox,
        )
        sample = stack({"optical": optical, "label": label})
        # One file per band, so each asset is one variable named by its key.
        files = sample.gs.to_cog(f"{root}/s{index}/{{collection}}/{{band}}.tif")
        tables.append(create_items(files, id=f"s{index}", datetime=WHEN))
    return open_vector(GeoVector.concat(tables).gs.to_geoparquet(root / "raw.parquet"))


def test_prepared_pixels_are_the_preprocessed_raw_pixels(tmp_path):
    spec = _spec()

    path = spec.prepare(_raw(tmp_path / "raw"), tmp_path / "prepared", on="label")

    prepared = open_vector(path)
    assert sorted(prepared["collection"].unique()) == ["image", "label"]
    assert ModelSpec.load(tmp_path / "prepared" / ModelSpec.filename) == spec
    windows = cuts.stacks(prepared, on="label")
    # Prepared items take their raw anchor's id as a prefix.
    assert windows["anchor"].tolist() == ["s0/label/label", "s1/label/label"]
    rasters = windows.iloc[1].gs.to_stack().gs.rasters
    assert list(rasters["image"].data_vars) == ["nir"]
    np.testing.assert_array_equal(
        rasters["image"]["nir"].values, 2 * _optical(1)["nir"].values
    )
    assert (rasters["label"]["label"].values == 1).all()


def test_pixels_stay_lazy_until_they_are_written(tmp_path, monkeypatch):
    raw = _raw(tmp_path / "raw")
    written = []
    to_cog = GeoStack.to_cog

    def spy(self, *args, **kwargs):
        written.append(
            all(
                isinstance(r[v].data, da.Array)
                for r in self.rasters.values()
                for v in r
            )
        )
        return to_cog(self, *args, **kwargs)

    monkeypatch.setattr(GeoStack, "to_cog", spy)

    _spec().prepare(raw, tmp_path / "prepared", on="label")

    assert written == [True, True]


def test_an_existing_output_is_refused(tmp_path):
    (tmp_path / "prepared").mkdir()

    with pytest.raises(FileExistsError):
        _spec().prepare(_raw(tmp_path / "raw"), tmp_path / "prepared", on="label")


def test_keep_given_as_text_is_refused(tmp_path):
    with pytest.raises(TypeError, match="keep"):
        _spec().prepare(
            _raw(tmp_path / "raw"), tmp_path / "prepared", on="label", keep="label"
        )
