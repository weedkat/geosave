"""A reusable geospatial observation flow through real local assets and storage."""

from geosave.geodata.stac.translate import table as stac_table

from datetime import UTC, datetime

import dask.array as da
import numpy as np
import pytest
from pystac.extensions.raster import RasterExtension

from geosave.geodata import open_raster
from geosave.geodata import attrs
from geosave.geodata.io.raster import stac as stac_io
from geosave.geodata.features import ndvi
from geosave.geodata import open_vector
from geosave.geodata.stac import create_catalog, create_items

from tests.geodata.conftest import build_raster


def test_described_rasters_round_trip_through_the_table(tmp_path) -> None:
    optical = build_raster(times=2, packed=True)[["red"]]
    optical = optical.gs.write_spectral("red", center_wavelength=0.665, target="red")
    optical = optical.assign(red=optical.red.assign_attrs(units="1"))
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    label = label.gs.write_legend(
        {0: "background", 1: "forest"}, {1: "#00ff00"}, target="label"
    )
    when = datetime(2025, 6, 1, tzinfo=UTC)
    optical_scenes = optical.gs.to_cog(
        f"{tmp_path}/optical_{{time:%Y%m%d}}_{{band}}.tif"
    )
    (label_scene,) = label.gs.to_cog(f"{tmp_path}/{{band}}.tif")
    rows = create_catalog(
        [
            *create_items(optical_scenes, id_format="optical_{start:%Y%m%d}"),
            *create_items(label_scene, id_format="label", datetime=when),
        ]
    )

    saved = rows.gs.to_geoparquet(tmp_path / "items.parquet")

    restored = stac_table.to_items(open_vector(saved))
    red = stac_io.open_raster([restored[0]]).red
    label = stac_io.open_raster([restored[-1]]).label
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"
    assert red.attrs["units"] == "1"
    assert red.gs.packing.scale_factor == pytest.approx(1e-4)
    assert red.gs.spectral.common_name == "red"
    assert red.gs.spectral.center_wavelength == 0.665
    assert red.gs.nodata == optical.red.gs.nodata
    legend = attrs.Legend.model_validate(label.attrs)
    assert legend.classes == {0: "background", 1: "forest"}


@pytest.mark.integration
def test_described_items_validate_against_their_schemas(tmp_path) -> None:
    scene = build_raster(times=1, packed=True)

    (written,) = scene.gs.to_cog(f"{tmp_path}/scene_{{time:%Y%m%d}}.tif")
    item = next(create_items(written, id_format="scene"))

    item.validate()


def test_observations_to_prepared_index_to_reusable_dataset(local_source, tmp_path):
    source, anchor = local_source
    raw = source.load(anchor)
    reflectance = raw.gs.mask_and_scale()
    derived = reflectance.assign(ndvi=ndvi(reflectance, nir="nir", red="red"))[["ndvi"]]
    composite = derived.median("time", keep_attrs=True)
    assert isinstance(composite.ndvi.data, da.Array)
    assert composite.gs.geobox == anchor.geobox
    written = composite.gs.to_zarr(tmp_path / "ndvi.zarr")

    with open_raster(written, chunks={}) as reopened:
        assert isinstance(reopened.ndvi.data, da.Array)
        assert reopened.gs.geobox == anchor.geobox
        assert np.isnan(reopened.ndvi.values[0, 0])
        assert reopened.ndvi.values[1, 1] == pytest.approx((1.1 - 0.2) / (1.1 + 0.2))
        assert "scale_factor" not in reopened.ndvi.attrs
    # Preparation leaves the reusable source configuration and packed input intact.
    assert raw.red.attrs["scale_factor"] == 0.0001
    assert raw.red.values[0, 1, 1] == 2000
