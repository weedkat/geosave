"""Items read back as the files and variable attrs they describe."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr
from pystac.extensions.eo import EOExtension

from geosave.geodata import attrs, stac
from geosave.geodata.io.stac import group_files_by_date, open_scene
from geosave.geodata.stac import band
from geosave.geodata.stac.catalog import from_items
from tests.geodata.conftest import build_raster, create_file_items


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    files = scene.gs.to_cog(tmp_path, path_format="forest_{time:%Y%m%d}_{band}.tif")
    entry = from_items(
        [next(create_file_items(files, id_format="forest"))]
    ).gs.to_items()[0]

    back = open_scene(entry).gs.to_dataset()
    assert sorted(back.gs.variables) == ["nir", "red"]
    assert back.gs.times[0] == pd.Timestamp("2025-06-01")


def _item(tmp_path, assets):
    """Build an Item whose data assets list EO bands, as `(key, bands)` pairs."""
    from datetime import UTC, datetime

    from pystac.extensions.eo import Band

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    EOExtension.add_to(item)
    for key, bands in assets:
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = [Band.create(**b) for b in bands]
    return item


def test_a_rendering_of_other_assets_bands_names_them_by_key(tmp_path) -> None:
    # Planetary Computer's shape: per-band files beside a true-colour rendering
    # whose eo:bands repeat their names, with no raster:bands count.
    item = _item(
        tmp_path,
        [
            ("B04", [{"name": "B04"}]),
            ("B03", [{"name": "B03"}]),
            ("B02", [{"name": "B02"}]),
            ("visual", [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}]),
        ],
    )

    files = stac.item.read_variable_names(item)

    assert list(files.values()) == [
        ("B04",),
        ("B03",),
        ("B02",),
        ("visual.1", "visual.2", "visual.3"),
    ]


def test_several_bands_of_one_file_take_the_names_they_state(tmp_path) -> None:
    # As odc-stac names them: the EO name, else the common name, else key.i.
    item = _item(
        tmp_path,
        [
            (
                "image",
                [{"name": "B02"}, {"name": None, "common_name": "nir"}, {"name": None}],
            )
        ],
    )

    (names,) = stac.item.read_variable_names(item).values()

    assert names == ("B02", "nir", "image.3")


def test_a_name_claimed_by_a_one_band_asset_falls_back_to_key(tmp_path) -> None:
    item = _item(
        tmp_path,
        [
            ("rgb", [{"name": "red"}, {"name": "green"}]),
            ("red", [{"name": "red"}]),
        ],
    )

    files = stac.item.read_variable_names(item)

    assert list(files.values()) == [
        ("rgb.1", "green"),
        ("red",),
    ]


def test_attrs_carry_band_facts_by_variable_name(tmp_path) -> None:
    from datetime import UTC, datetime

    from pystac.extensions.eo import Band

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    EOExtension.add_to(item)
    listed = {
        "B04": [Band.create(name="B04", common_name="red", center_wavelength=0.665)],
        "visual": [Band.create(name=n) for n in ("B04", "B03", "B02")],
    }
    for key, bands in listed.items():
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = bands

    red = band.read_attrs(xr.DataArray(0), item.assets["B04"], 0)
    visual = band.read_attrs(xr.DataArray(0), item.assets["visual"], 0)
    assert attrs.Spectral.from_attrs(red).common_name == "red"
    assert visual == {}


def test_scene_combines_instant_and_composite_coverage(tmp_path):
    from geosave.geodata.transform import temporal
    from tests.geodata.conftest import build_raster

    instant = build_raster(times=1).isel(time=0)
    monthly = temporal.reduce(temporal.resample(build_raster(times=1), "MS"), "median")
    composite = monthly.isel(time=0)
    files = {
        "instant": (tmp_path / "instant.tif", instant),
        "composite": (tmp_path / "composite.tif", composite),
    }

    item = stac.item.create_item(files, id_format="{start:%Y%m%d}_{lat:.4f}_{lon:.4f}")

    assert item.datetime.isoformat() == "2025-06-01T00:00:00+00:00"
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-06-30")


def test_scene_reads_explicit_coverage_from_each_file(tmp_path):

    from tests.geodata.conftest import build_raster

    instant = build_raster(times=1).isel(time=0)
    composite = instant.assign_coords(
        start_datetime=np.datetime64("2025-05-15"),
        end_datetime=np.datetime64("2025-06-15"),
    )
    files = {
        "instant": (tmp_path / "instant.tif", instant),
        "composite": (tmp_path / "composite.tif", composite),
    }

    item = stac.item.create_item(files, id_format="{start:%Y%m%d}_{lat:.4f}_{lon:.4f}")

    assert item.datetime.isoformat() == "2025-06-01T00:00:00+00:00"
    assert item.properties["start_datetime"].startswith("2025-05-15")
    assert item.properties["end_datetime"].startswith("2025-06-15")


def test_composite_keeps_nominal_time_and_bounds(tmp_path):
    from geosave.geodata.transform import temporal

    ds = temporal.reduce(temporal.resample(build_raster(times=1), "MS"), "median")
    scene = ds.gs.to_cog(tmp_path, path_format="composite.tif")[0]
    item = next(create_file_items(scene))
    assert item.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-06-30")


def test_unsplit_multidate_store_cannot_be_one_observation(tmp_path):
    ds = build_raster(times=2)
    store = ds.gs.to_netcdf(tmp_path / "cube.nc")
    with pytest.raises(ValueError, match="split the dates"):
        list(create_file_items(store))
    split = ds.gs.to_cog(tmp_path, path_format="date-{time:%Y%m%d}.tif")
    assert len(list(create_file_items(split))) == 2


def test_coordinates_are_not_properties_by_their_name(tmp_path):
    ds = build_raster(times=1).assign_coords({"eo:cloud_cover": 12.0, "platform": "x"})
    item = next(create_file_items(ds.gs.to_cog(tmp_path, path_format="named.tif")))
    assert "eo:cloud_cover" not in item.properties
    assert "platform" not in item.properties


def test_an_item_takes_the_common_properties_overlaid_with_its_dates(tmp_path):
    ds = build_raster(times=1)
    ds = attrs.stamp_attrs(
        ds,
        attrs.StacMetadata(
            common_properties={
                "platform": "sentinel-2a",
                "constellation": "sentinel-2",
            },
            extensions=["https://stac-extensions.github.io/view/v1.0.0/schema.json"],
        ),
    )
    ds = stac.item.stamp_properties_coordinate(
        ds, [{"platform": "sentinel-2b", "view:sun_azimuth": 150.0}]
    )

    item = stac.item.create_item(
        {"image": (tmp_path / "scene.tif", ds)}, id_format="scene"
    )

    assert item.properties["platform"] == "sentinel-2b"
    assert item.properties["constellation"] == "sentinel-2"
    assert item.properties["view:sun_azimuth"] == 150.0
    assert (
        "https://stac-extensions.github.io/view/v1.0.0/schema.json"
        in item.stac_extensions
    )


def test_carried_properties_never_restate_what_the_raster_states(tmp_path):
    ds = stac.item.stamp_properties_coordinate(
        build_raster(times=1),
        [{"start_datetime": "1999-01-01T00:00:00Z", "proj:code": "EPSG:4326"}],
    )

    item = stac.item.create_item(
        {"image": (tmp_path / "scene.tif", ds)}, id_format="scene"
    )

    assert "start_datetime" not in item.properties
    assert "proj:code" not in item.properties


def test_the_properties_coordinate_round_trips_each_date(tmp_path):
    ds = build_raster(times=2)
    mappings = [{"eo:cloud_cover": 20.0, "instruments": ["msi"]}, {"platform": "s2b"}]

    recorded = stac.item.stamp_properties_coordinate(ds, mappings)
    recorded.to_zarr(tmp_path / "scene.zarr")
    restored = xr.open_zarr(tmp_path / "scene.zarr")

    assert recorded.stac_properties.dims == ("time",)
    assert stac.item.read_properties_coordinate(restored) == mappings
    assert stac.item.read_properties_coordinate(restored.isel(time=1)) == [
        {"platform": "s2b"}
    ]
    with pytest.raises(ValueError, match="2 time labels"):
        stac.item.stamp_properties_coordinate(ds, [{}])


def test_observation_properties_leave_out_what_a_raster_states():
    kept = stac.item.select_item_properties(
        {
            "datetime": "2025-01-15T00:00:00Z",
            "proj:shape": [10, 10],
            "eo:bands": [],
            "eo:cloud_cover": 4.2,
            "platform": "sentinel-2a",
        }
    )
    assert kept == {"eo:cloud_cover": 4.2, "platform": "sentinel-2a"}


def test_two_files_of_one_date_cannot_hold_one_asset(tmp_path):
    ds = build_raster(times=1)[["red"]]
    with pytest.raises(ValueError, match="asset 'red'"):
        group_files_by_date({tmp_path / "a.tif": ds, tmp_path / "b.tif": ds})


def test_equal_explicit_interval_is_an_instant(tmp_path):
    ds = build_raster()
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    item = next(
        create_file_items(
            ds.gs.to_cog(tmp_path, path_format="undated.tif"), datetime=(stamp, stamp)
        )
    )
    assert item.datetime == stamp


def test_item_identity_uses_union_footprint_and_grid_fields_require_one_grid(tmp_path):

    ds = build_raster(times=1).isel(time=0)
    shifted = ds.assign_coords(x=ds.x + 1000)
    files = {
        "first": (tmp_path / "first.tif", ds),
        "second": (tmp_path / "second.tif", shifted),
    }
    first = stac.item.create_item(files, id_format="{start:%Y%m%d}_{lat:.4f}_{lon:.4f}")
    second = stac.item.create_item(
        dict(reversed(list(files.items()))),
        id_format="{start:%Y%m%d}_{lat:.4f}_{lon:.4f}",
    )
    assert (first.id, first.geometry) == (second.id, second.geometry)
    with pytest.raises(ValueError, match="one grid"):
        stac.item.create_item(files, id_format="{width}")


def test_projected_item_id_keeps_geographic_centroid_and_native_grid(tmp_path):
    from odc.geo.geobox import GeoBox
    from shapely.geometry import shape

    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox(
        (0, 5_000_000, 1_000_000, 8_000_000), crs="EPSG:3857", resolution=100_000
    )
    raster = build_dataset(
        {"red": (("y", "x"), np.zeros(tuple(grid.shape), dtype="uint8"))}, grid
    ).assign_coords(time=np.datetime64("2025-06-01"))
    entry = stac.item.create_item(
        {"red": (tmp_path / "red.tif", raster)},
        id_format="{lat:.4f}_{lon:.4f}_{epsg}_{res:.0f}_{width:.0f}_{start:%Y%m%d}",
    )

    center = shape(entry.geometry).centroid
    assert entry.id == f"{center.y:.4f}_{center.x:.4f}_3857_100000_1000000_20250601"
    assert abs(center.y - raster.gs.anchor.lonlat[1]) > 0.1
