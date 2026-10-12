"""Items built from the rasters a writer saved."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension

from geosave.geodata import io, raster, stac, stack
from geosave.geodata import open_vector
from geosave.geodata.io.vector import stac_geoparquet as table

from tests.geodata.conftest import build_raster, cog_template, stack_templates

IDS = ["forest_20250601T000000", "forest_20250602T000000"]


def _saved(data, root, name, *, by_band=False):
    """Write a raster or stack in the layout these Items are named after."""
    if isinstance(data, xr.DataTree):
        return data.gs.to_cog(stack_templates(data, root / name))
    template = cog_template(root, name, dated="time" in data.dims)
    if by_band:
        template = template.removesuffix(".tif") + "/{band}.tif"
    return data.gs.to_cog(template)


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    source = build_raster(times=2)
    for built in table.to_items(stac.create_items(_saved(source, tmp_path, "forest"))):
        built.validate()


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))}, grid
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    (built,) = table.to_items(stac.create_item({"label": path}, id="laea"))

    assert built.bbox[0] == pytest.approx(56.78, abs=0.01)
    assert built.bbox[1] == pytest.approx(12.34, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    source = build_raster(times=2)
    rows = stac.create_items(_saved(source, tmp_path, "forest", by_band=True))

    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = table.to_items(open_vector(saved))
    assert [entry.id for entry in restored] == IDS
    assert sorted(restored[0].assets) == ["nir", "red"]
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"


def test_one_file_per_scene_is_one_item_per_instant(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = table.to_items(stac.create_items(paths))

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert [list(each.assets) for each in built] == [["image"], ["image"]]
    assert built[1].assets["image"].href == paths[1].href
    assert built[1].datetime == datetime(2025, 6, 11, tzinfo=UTC)
    assert [each.datetime.isoformat() for each in built] == [
        "2025-06-01T00:00:00+00:00",
        "2025-06-11T00:00:00+00:00",
    ]
    assert all("start_datetime" not in each.properties for each in built)


def test_split_bands_become_one_asset_per_variable(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest", by_band=True)

    built = table.to_items(stac.create_items(paths))

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert sorted(built[0].assets) == ["nir", "red"]
    assert built[1].assets["nir"].href == paths[3].href


def test_items_take_the_id_of_their_collection_without_a_link(scene, tmp_path):
    forest = stac.create_collection("forest", description="Forest samples")

    built = table.to_items(
        stac.create_items(_saved(scene, tmp_path, "forest"), collection=forest)
    )

    assert [each.collection_id for each in built] == ["forest", "forest"]
    assert [each.links for each in built] == [[], []]


def test_two_files_claiming_one_asset_key_refuse(scene, tmp_path):
    (first,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/a_{{time:%Y%m%d}}.tif")
    (second,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/b_{{time:%Y%m%d}}.tif")

    with pytest.raises(ValueError, match="both be the asset 'image'"):
        stac.create_items([first, second])


def test_an_item_states_what_the_file_stores(tmp_path):
    source = build_raster(times=2)[["red"]]
    path = tmp_path / "small.tif"
    # A GeoTIFF written by another tool stores the type it was told to.
    source.red.isel(time=0).rio.to_raster(path, dtype="uint8")

    (built,) = table.to_items(
        stac.create_item(
            {"red": path}, id="small", datetime=datetime(2025, 6, 1, tzinfo=UTC)
        )
    )

    assert source.red.dtype == "uint16"
    assert RasterExtension.ext(built.assets["red"]).bands[0].data_type == "uint8"


def test_no_paths_refuse():
    with pytest.raises(ValueError, match="records to_cog returned"):
        stac.create_items([])


def test_an_id_template_is_filled_per_scene(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = table.to_items(stac.create_items(paths, id="{lat:.2f}N_{start:%Y%m%d}"))

    assert [each.id for each in built] == ["45.13N_20250601", "45.13N_20250611"]
    assert built[0].assets["image"].href == paths[0].href


def test_an_id_template_that_repeats_refuses(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    with pytest.raises(ValueError, match="same id"):
        stac.create_items(paths, id="{lat:.2f}N")


def test_building_items_computes_no_pixels(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    with dask.config.set(scheduler=_refuse):
        built = stac.create_items(paths, chunks={})

    assert len(built) == 2


def test_a_timeless_raster_needs_a_time(label, tmp_path):
    files = _saved(label, tmp_path, "label")

    with pytest.raises(ValueError, match="pass datetime="):
        stac.create_items(files)

    (built,) = table.to_items(
        stac.create_items(files, datetime=datetime(2025, 6, 1, tzinfo=UTC))
    )
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_the_caller_names_the_assets_of_one_item(scene, label, tmp_path):
    (optical, _) = (file.href for file in _saved(scene, tmp_path, "optical"))
    (classes,) = (file.href for file in _saved(label, tmp_path, "label"))

    (built,) = table.to_items(
        stac.create_item({"optical": optical, "label": classes}, id="s0")
    )

    assert built.id == "s0"
    assert sorted(built.assets) == ["label", "optical"]
    assert built.assets["label"].href == str(classes)
    # The timeless label rides on the instant its dated sibling covers.
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_a_named_asset_cannot_be_a_datacube_store(scene, tmp_path) -> None:
    store = scene.gs.to_zarr(tmp_path / "cube.zarr")

    with pytest.raises(ValueError, match="to_cog"):
        stac.create_item({"image": store}, id="cube")


def test_a_label_item_declares_only_what_it_uses(label, tmp_path: Path) -> None:
    classes = label.gs.write_legend({1: "forest"}, target="label")
    (path,) = (file.href for file in _saved(classes, tmp_path, "label"))

    (built,) = table.to_items(
        stac.create_item(
            {"label": path}, id="label", datetime=datetime(2025, 1, 1, tzinfo=UTC)
        )
    )

    assert set(built.stac_extensions) == {
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        EOExtension.get_schema_uri(),
        ClassificationExtension.get_schema_uri(),
    }


def test_a_plain_scene_declares_its_grid_bands_and_time(scene, tmp_path: Path) -> None:
    (path, _) = (file.href for file in _saved(scene, tmp_path, "scene"))

    (built,) = table.to_items(stac.create_item({"image": path}, id="scene"))

    assert built.stac_extensions == [
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        EOExtension.get_schema_uri(),
    ]


def test_a_collection_starts_with_an_open_extent() -> None:
    forest = stac.create_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )

    assert (forest.id, forest.license) == ("forest", "CC-BY-4.0")
    assert forest.extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]
    assert forest.extent.temporal.intervals == [[None, None]]


def test_every_extension_module_writes_onto_each_asset(scene, tmp_path) -> None:
    (path, _) = (file.href for file in _saved(scene, tmp_path, "scene"))
    (built,) = table.to_items(stac.create_item({"image": path}, id="scene"))

    asset = built.assets["image"]
    assert ProjectionExtension.ext(asset).code == "EPSG:32633"
    assert len(RasterExtension.ext(asset).bands) == 2
    assert asset.media_type == "image/tiff; application=geotiff"
    assert asset.roles == ["data"]


def test_odc_stac_names_a_multiband_asset_after_its_bands(tmp_path: Path) -> None:
    import odc.stac

    scene = build_raster()
    (built,) = table.to_items(
        stac.create_items(
            _saved(scene, tmp_path, "forest"),
            datetime=datetime(2025, 6, 1, tzinfo=UTC),
        )
    )

    loaded = odc.stac.load([built], geobox=scene.gs.geobox, chunks={})

    assert list(loaded.data_vars) == ["red", "nir"]


def test_create_item_returns_one_row(label, tmp_path: Path) -> None:
    (path,) = (file.href for file in _saved(label, tmp_path, "label"))

    rows = stac.create_item(
        {"label": path}, id="s0", datetime=datetime(2025, 6, 1, tzinfo=UTC)
    )

    assert rows["id"].tolist() == ["s0"]
    assert list(rows["assets"].iloc[0]) == ["label"]


def test_records_become_one_item_per_scene(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")

    rows = stac.create_items(files)

    assert len(rows) == 2
    item = table.to_items(rows)[0]
    assert sorted(item.assets) == ["nir", "red"]
    assert [b.name for b in EOExtension.ext(item.assets["red"]).bands] == ["red"]


def test_a_multiband_file_is_one_image_asset_with_bands_in_order(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    item = table.to_items(stac.create_items(files))[0]

    assert list(item.assets) == ["image"]
    names = [b.name for b in EOExtension.ext(item.assets["image"]).bands]
    assert names == ["red", "nir"]


def test_a_stacks_layers_become_collections(scene, label, tmp_path) -> None:
    files = stack({"optical": scene, "label": label}).gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    rows = stac.create_items(files, id="s0")

    assert rows["collection"].tolist() == ["optical", "optical", "label"]
    assert rows["id"].tolist()[2] == "s0/label"
    assert str(rows.iloc[2]["start_datetime"])[:10] == "2025-06-01"


def test_a_collection_cannot_override_named_layers(scene, tmp_path) -> None:
    files = stack({"optical": scene}).gs.to_cog(
        f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif"
    )

    with pytest.raises(ValueError, match="already name"):
        stac.create_items(
            files, collection=stac.create_collection("x", description="x")
        )


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    from geosave.geodata.stac.item import cog_files

    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")
    item = table.to_items(stac.create_items(files))[0]

    back = cog_files(item)

    assert sorted(f.bands for f in back) == [("nir",), ("red",)]
    assert {f.datetime for f in back} == {files[0].datetime}


def test_paths_are_not_records(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    with pytest.raises(TypeError, match="CogFile"):
        stac.create_items([file.href for file in files])


def test_a_collection_can_be_named_by_its_id(scene, tmp_path):
    (path,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    one = stac.create_item({"image": path.href}, id="s0", collection="image")
    many = stac.create_items([path], collection="image")

    assert one["collection"].tolist() == ["image"]
    assert many["collection"].tolist() == ["image"]


def test_records_name_variables_by_asset_key(tmp_path) -> None:
    # Planetary Computer's shape: per-band files beside a true-colour rendering
    # whose eo:bands repeat their names, with no raster:bands count.
    from datetime import UTC, datetime

    import pystac
    from pystac.extensions.eo import Band

    from geosave.geodata.stac.item import cog_files

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

    records = cog_files(item)

    assert [record.bands for record in records] == [
        ("B04",),
        ("visual.1", "visual.2", "visual.3"),
    ]


def test_band_attrs_carry_eo_facts_by_variable_name(tmp_path) -> None:
    from datetime import UTC, datetime

    import pystac
    from pystac.extensions.eo import Band

    import geosave.geodata.attrs as attrs
    from geosave.geodata.stac.item import band_attrs

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

    found = band_attrs(item)

    assert attrs.Spectral.from_attrs(found["B04"]).common_name == "red"
    # visual's bands state only a name, which names no CF attr.
    assert set(found) == {"B04"}
