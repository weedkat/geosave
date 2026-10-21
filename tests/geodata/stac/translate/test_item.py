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
from geosave.geodata.io.raster.stac import open_raster
from geosave.geodata.stac import translate
from geosave.geodata.stac.translate import band
from geosave.geodata.stac.translate import table as stac_table
from tests.geodata.conftest import build_raster


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")
    entry = stac_table.to_items(
        stac.create_catalog([next(stac.create_items(files[0], id_format="forest"))])
    )[0]

    back = open_raster([entry])
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

    files = translate.item.read_variable_names(item)

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

    (names,) = translate.item.read_variable_names(item).values()

    assert names == ("B02", "nir", "image.3")


def test_a_name_claimed_by_a_one_band_asset_falls_back_to_key(tmp_path) -> None:
    item = _item(
        tmp_path,
        [
            ("rgb", [{"name": "red"}, {"name": "green"}]),
            ("red", [{"name": "red"}]),
        ],
    )

    files = translate.item.read_variable_names(item)

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
    from geosave.geodata.attrs import TimeSpec
    from geosave.geodata.io.raster.file import RasterFile, Scene
    from tests.geodata.conftest import build_raster

    instant = build_raster(times=1).isel(time=0)
    composite = instant.copy(deep=True)
    composite.time.attrs.update(TimeSpec.from_resample("MS").to_attrs())
    files = {
        "instant": RasterFile.from_xarray(tmp_path / "instant.tif", instant),
        "composite": RasterFile.from_xarray(tmp_path / "composite.tif", composite),
    }

    item = next(stac.create_items(Scene(None, files)))

    assert item.datetime.isoformat() == "2025-06-01T00:00:00+00:00"
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-06-30")


def test_scene_reads_explicit_coverage_from_each_file(tmp_path):

    from geosave.geodata.io.raster.file import RasterFile, Scene
    from tests.geodata.conftest import build_raster

    instant = build_raster(times=1).isel(time=0)
    composite = instant.assign_coords(
        start_datetime=np.datetime64("2025-05-15"),
        end_datetime=np.datetime64("2025-06-15"),
    )
    files = {
        "instant": RasterFile.from_xarray(tmp_path / "instant.tif", instant),
        "composite": RasterFile.from_xarray(tmp_path / "composite.tif", composite),
    }

    item = next(stac.create_items(Scene(None, files)))

    assert item.datetime.isoformat() == "2025-06-01T00:00:00+00:00"
    assert item.properties["start_datetime"].startswith("2025-05-15")
    assert item.properties["end_datetime"].startswith("2025-06-15")


def test_composite_keeps_nominal_time_and_bounds(tmp_path):
    from geosave.geodata.attrs import TimeSpec

    ds = build_raster(times=1)
    ds.time.attrs.update(TimeSpec.from_resample("MS").to_attrs())
    scene = ds.gs.to_cog(str(tmp_path / "composite.tif"))[0]
    item = next(stac.create_items(scene))
    assert item.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-06-30")


def test_unsplit_multidate_store_cannot_be_one_observation(tmp_path):
    ds = build_raster(times=2)
    scenes = ds.gs.to_netcdf(tmp_path / "cube.nc")
    with pytest.raises(ValueError, match="split multiple dates"):
        list(stac.create_items(scenes))
    split = [
        ds.isel(time=index).gs.to_netcdf(tmp_path / f"date-{index}.nc")
        for index in range(2)
    ]
    assert len(list(stac.create_items(split))) == 2


@pytest.mark.parametrize("value", [-1.0, 101.0, float("inf")])
def test_invalid_cloud_cover_rejected(tmp_path, value):
    ds = build_raster(times=1).assign_coords({"eo:cloud_cover": value})
    scenes = ds.gs.to_cog(tmp_path / "scene.tif")
    with pytest.raises(ValueError, match="eo:cloud_cover"):
        list(stac.create_items(scenes))


def test_missing_and_dropped_properties_are_not_exported(tmp_path):
    ds = build_raster(times=1).assign_coords(
        {"eo:cloud_cover": float("nan"), "unknown:value": 12}
    )
    item = next(stac.create_items(ds.gs.to_cog(tmp_path / "missing.tif")))
    assert "eo:cloud_cover" not in item.properties
    assert "unknown:value" not in item.properties
    ds = ds.assign_coords({"eo:cloud_cover": 12}).drop_vars("eo:cloud_cover")
    item = next(stac.create_items(ds.gs.to_cog(tmp_path / "dropped.tif")))
    assert "eo:cloud_cover" not in item.properties


def test_item_id_and_geometry_do_not_depend_on_asset_order(tmp_path):
    from geosave.geodata.io.raster.file import Scene

    ds = build_raster(times=1)
    scene = ds.gs.to_cog(str(tmp_path / "{band}.tif"))[0]
    reversed_scene = Scene(
        scene.collection,
        dict(reversed(list(scene.written.items()))),
        coords=scene.coords,
    )
    first = next(stac.create_items(scene))
    second = next(stac.create_items(reversed_scene))
    assert (first.id, first.geometry) == (second.id, second.geometry)


def test_equal_explicit_interval_is_an_instant(tmp_path):
    ds = build_raster()
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    item = next(
        stac.create_items(
            ds.gs.to_cog(tmp_path / "undated.tif"), datetime=(stamp, stamp)
        )
    )
    assert item.datetime == stamp


def test_item_identity_uses_union_footprint_and_grid_fields_require_one_grid(tmp_path):
    from geosave.geodata.io.raster.file import RasterFile, Scene

    ds = build_raster(times=1).isel(time=0)
    shifted = ds.assign_coords(x=ds.x + 1000)
    files = {
        "first": RasterFile.from_xarray(tmp_path / "first.tif", ds),
        "second": RasterFile.from_xarray(tmp_path / "second.tif", shifted),
    }
    first = next(stac.create_items(Scene(None, files)))
    second = next(stac.create_items(Scene(None, dict(reversed(list(files.items()))))))
    assert (first.id, first.geometry) == (second.id, second.geometry)
    with pytest.raises(ValueError, match="common grid"):
        next(stac.create_items(Scene(None, files), id_format="{width}"))
