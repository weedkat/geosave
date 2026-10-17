"""Native STAC Items from other producers fit the item table."""

from geosave.geodata.stac.translate import table as stac_table

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pystac
import pytest
import numpy as np
from shapely.geometry import mapping
from stac_geoparquet.arrow import stac_table_to_items

from geosave.geodata import io
from geosave.geodata import open_vector
from tests.geodata.conftest import build_raster, read_items


def _item(tmp_path, name="scene"):
    source = build_raster(times=1).isel(time=0)
    saved = io.geotiff.write_cog(source, tmp_path / f"{name}.tif")
    shape = source.gs.geobox.extent.to_crs("EPSG:4326").geom
    item = pystac.Item(
        name,
        mapping(shape),
        list(shape.bounds),
        datetime(2025, 6, 1, tzinfo=UTC),
        {"eo:cloud_cover": 12.5, "provider:quality": "good"},
    )
    item.add_asset(
        "reflectance",
        pystac.Asset(str(saved), media_type=pystac.MediaType.COG, roles=["data"]),
    )
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://example.com/thumb.jpg",
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )
    return item, source


def test_a_native_item_needs_no_geosave_fields(tmp_path):
    entry, _ = _item(tmp_path)

    rows = stac_table.from_items([entry])

    assert rows.id.tolist() == [entry.id]
    assert rows.iloc[0]["provider:quality"] == "good"
    assert not any(name.startswith("geosave:") for name in rows.columns)
    assert (
        rows.iloc[0].assets["reflectance"]["href"] == entry.assets["reflectance"].href
    )


def test_relative_assets_resolve_without_mutating_the_item(tmp_path):
    entry, _ = _item(tmp_path)
    entry.assets["reflectance"].href = "scene.tif"
    entry.set_self_href(str(tmp_path / "scene.json"))

    rows = stac_table.from_items([entry])

    assert entry.assets["reflectance"].href == "scene.tif"
    assert rows.iloc[0].assets["reflectance"]["href"] == str(tmp_path / "scene.tif")


def test_a_written_table_retains_item_and_asset_properties(tmp_path):
    first, source = _item(tmp_path, "first")
    second, _ = _item(tmp_path, "second")
    bands = [
        {"name": name, "data_type": source[name].dtype.name}
        for name in source.gs.variables
    ]
    first.assets["reflectance"].extra_fields["bands"] = bands
    second.properties["new:property"] = 7
    second.assets["other"] = second.assets.pop("reflectance")

    saved = stac_table.from_items([second, first]).gs.to_geoparquet(
        tmp_path / "items.parquet"
    )

    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_only_data_assets_load(tmp_path):
    entry, source = _item(tmp_path)

    loaded = read_items(stac_table.from_items([entry]))

    # The thumbnail is skipped; the data asset names its bands by its key.
    assert list(loaded.data_vars) == [
        f"reflectance.{index}" for index in range(1, len(source.data_vars) + 1)
    ]
    loaded.close()


def test_a_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    entry, _ = _item(tmp_path)
    first = stac_table.from_items([entry]).gs.to_geoparquet(
        tmp_path / "catalog.parquet"
    )
    second = open_vector(first).gs.to_geoparquet(tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    entry, _ = _item(tmp_path)
    rows = stac_table.from_items([entry])
    rows = rows.set_geometry(rows.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(rows.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)


@pytest.mark.parametrize("folder", ["dataset", "data set"])
def test_a_table_follows_its_assets_when_moved(tmp_path, folder):
    home = tmp_path / folder
    home.mkdir()
    entry, _ = _item(home)
    saved = stac_table.from_items([entry]).gs.to_geoparquet(home / "catalog.parquet")
    stored = pq.read_table(saved).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = open_vector(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()


def test_thumbnail_cannot_change_sample_grid_or_data_names(tmp_path):
    from geosave.geodata.stac.extensions import projection

    entry, source = _item(tmp_path)
    entry.collection_id = "optical"
    data = entry.assets["reflectance"]
    data.extra_fields.update(projection.from_geobox(source.gs.geobox))
    data.extra_fields["bands"] = [{"name": name} for name in source.data_vars]
    thumbnail = entry.assets["thumbnail"]
    thumbnail.extra_fields.update(projection.from_geobox(source.gs.geobox[:1, :1]))
    entry.assets = {"red": thumbnail, "reflectance": data}

    samples = stac_table.from_items([entry]).gs.sample()
    row = samples.iloc[0]
    loaded = row.gs.to_xarray()["optical"].dataset

    assert row.gs.geobox == source.gs.geobox
    assert list(loaded.data_vars) == list(source.data_vars)
    assert loaded.red.chunks is not None
    assert loaded.gs.geobox == source.gs.geobox
    np.testing.assert_array_equal(loaded.red.values, source.red.values)


def test_unreferenced_image_remains_readable_without_a_sampling_grid(tmp_path):
    import dask.array as da
    import rasterio

    from geosave.geodata.warnings import MissingGridWarning

    path = tmp_path / "plain.tif"
    pixels = np.arange(12, dtype="uint8").reshape(3, 4)
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        with rasterio.open(
            path, "w", driver="GTiff", height=3, width=4, count=1, dtype="uint8"
        ) as output:
            output.write(pixels, 1)
    item = pystac.Item(
        "plain",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        [0, 0, 1, 1],
        datetime(2025, 1, 1, tzinfo=UTC),
        {},
        collection="images",
    )
    item.add_asset(
        "image",
        pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF, roles=["data"]),
    )
    rows = stac_table.from_items([item])

    with pytest.warns(MissingGridWarning, match="plain"):
        assert rows.iloc[0].gs.anchor is None
    with pytest.warns(MissingGridWarning, match="plain"):
        assert rows.gs.sample().empty
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        loaded = rows.iloc[0].gs.to_xarray()

    assert loaded.gs.crs is None
    assert isinstance(loaded.image.data, da.Array)
    np.testing.assert_array_equal(loaded.image.values, pixels)
