"""Items read back as the files and variable attrs they describe."""

from __future__ import annotations

import pandas as pd
from pystac.extensions.eo import EOExtension

from geosave.geodata import stac
from geosave.geodata.io.vector import stac_geoparquet as table
from geosave.geodata.stac import translate


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")
    entry = table.to_item(stac.create_item(files))[0]

    back = translate.item.files(entry)

    assert sorted(f.bands for f in back) == [("nir",), ("red",)]
    assert {f.datetime for f in back} == {pd.Timestamp("2025-06-01")}


def test_records_name_variables_by_asset_key(tmp_path) -> None:
    # Planetary Computer's shape: per-band files beside a true-colour rendering
    # whose eo:bands repeat their names, with no raster:bands count.
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
    for key, names in (("B04", ["B04"]), ("visual", ["B04", "B03", "B02"])):
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = [Band.create(name=n) for n in names]

    records = translate.item.files(item)

    assert [record.bands for record in records] == [
        ("B04",),
        ("visual.1", "visual.2", "visual.3"),
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

    found = translate.item.attrs(item)

    assert attrs.Spectral.from_attrs(found["B04"]).common_name == "red"
    # visual's bands state only a name, which names no CF attr.
    assert set(found) == {"B04"}


def test_attrs_read_every_schema_from_core_bands(tmp_path) -> None:
    from datetime import UTC, datetime

    import pystac

    import geosave.geodata.attrs as attrs

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    band = {
        "name": "label",
        "unit": "1",
        "raster:scale": 0.5,
        "eo:common_name": "red",
        "classification:classes": [{"value": 1, "name": "forest"}],
    }
    item.add_asset(
        "label",
        pystac.Asset(f"{tmp_path}/label.tif", roles=["data"], extra_fields={"bands": [band]}),
    )

    found = translate.item.attrs(item)["label"]

    assert found["units"] == "1"
    assert attrs.Packing.from_attrs(found) == attrs.Packing(scale_factor=0.5)
    assert attrs.Spectral.from_attrs(found).common_name == "red"
    assert attrs.Legend.from_attrs(found).classes == {1: "forest"}


def test_files_of_one_collection_instant_and_grid_are_one_scene(scene, tmp_path) -> None:
    from geosave.geodata import open_raster

    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")
    headers = {str(file): open_raster(file) for file in files}

    found = translate.item.scenes(headers)

    assert [scene.stem for scene in found] == ["forest_20250601", "forest_20250611"]
    assert [sorted(scene.assets) for scene in found] == [["nir", "red"], ["nir", "red"]]


def test_a_file_is_the_asset_of_its_one_band_or_an_image(scene, tmp_path) -> None:
    from geosave.geodata import open_raster

    (both, _) = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    (red, *_) = scene[["red"]].gs.to_cog(f"{tmp_path}/r_{{time:%Y%m%d}}.tif")

    found = translate.item.scenes(
        {str(both): open_raster(both), str(red): open_raster(red)}
    )

    # One date on one grid is one scene; each file is the asset its bands make.
    assert [list(scene.assets) for scene in found] == [["image", "red"]]


def test_scenes_take_each_files_stated_collection_unless_given_one(scene, tmp_path):
    from geosave.geodata import attrs, open_raster

    named = scene.gs.write_acdd(attrs.ACDD(id="sentinel-2-l2a"))
    (path, _) = named.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    headers = {str(path): open_raster(path)}

    assert translate.item.scenes(headers)[0].collection == "sentinel-2-l2a"
    assert translate.item.scenes(headers, collection="image")[0].collection == "image"


def test_a_lone_file_names_its_scene() -> None:
    assets = {"label": "/data/s0/label.tif"}

    assert translate.item.Scene(None, None, None, assets).stem == "label"
