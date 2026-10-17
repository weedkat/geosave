"""Items read back as the files and variable attrs they describe."""

from __future__ import annotations

from geosave.geodata.stac.translate import table as stac_table

import pandas as pd
from pystac.extensions.eo import EOExtension

from geosave.geodata import stac
from geosave.geodata.stac import translate
from geosave.geodata.io.raster.stac import open_raster
from geosave.geodata.stac.translate import band
import xarray as xr


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

    import pystac
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

    import pystac
    from pystac.extensions.eo import Band

    import geosave.geodata.attrs as attrs

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
    import numpy as np

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
