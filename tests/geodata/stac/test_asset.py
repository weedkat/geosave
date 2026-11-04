"""Raster file descriptions become native Assets without reading pixels."""

from datetime import UTC, datetime

import dask
import pystac

from geosave.geodata.stac import asset
from tests.geodata.conftest import build_raster, create_file_items


def test_file_translation_returns_owned_asset_without_assembling_item(tmp_path):
    raster = build_raster(times=1).isel(time=0).chunk()
    path = tmp_path / "image.tif"
    owner = pystac.Item("scene", None, None, datetime(2025, 6, 1, tzinfo=UTC), {})

    def refuse_pixels(*args, **kwargs):
        raise AssertionError("translation computed pixels")

    with dask.config.set(scheduler=refuse_pixels):
        result = asset.build_asset(path, raster, owner=owner)

    assert owner.assets == {}
    assert result.owner is owner
    assert result.href == str(tmp_path / "image.tif")
    assert result.roles == ["data"]
    assert result.extra_fields["proj:shape"] == list(raster.gs.geobox.shape)
    assert len(result.extra_fields["raster:bands"]) == len(raster.data_vars)
    assert any("projection" in schema for schema in owner.stac_extensions)
    assert any("raster" in schema for schema in owner.stac_extensions)

    owner.add_asset("image", result)
    assert owner.assets["image"] is result


def test_an_asset_reads_back_as_the_header_written(tmp_path):
    from geosave.geodata import attrs
    from geosave.geodata.stac import band

    raster = build_raster(times=1, packed=True).isel(time=0)
    scene = raster.gs.to_cog(tmp_path, path_format="image.tif")[0]
    item = next(create_file_items(scene))
    entry = item.assets["image"]

    assert entry.href == str(scene)
    for index, name in enumerate(raster.data_vars):
        metadata = band.read_attrs(raster[name], entry, index)
        assert attrs.Packing.from_attrs(metadata) == raster[name].gs.packing
