"""Collections and Items built from the rasters a writer saved."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dask
import numpy as np
import pystac
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
    for built in table.to_item(stac.create_item(_saved(source, tmp_path, "forest"))):
        built.validate()


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))}, grid
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    (built,) = table.to_item(stac.create_item({"label": path}, id="laea"))

    assert built.bbox[0] == pytest.approx(56.78, abs=0.01)
    assert built.bbox[1] == pytest.approx(12.34, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    source = build_raster(times=2)
    rows = stac.create_item(_saved(source, tmp_path, "forest", by_band=True))

    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = table.to_item(open_vector(saved))
    assert [entry.id for entry in restored] == IDS
    assert sorted(restored[0].assets) == ["nir", "red"]
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"


def test_one_file_per_scene_is_one_item_per_instant(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = table.to_item(stac.create_item(paths))

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert [list(each.assets) for each in built] == [["image"], ["image"]]
    assert built[1].assets["image"].href == str(paths[1])
    assert built[1].datetime == datetime(2025, 6, 11, tzinfo=UTC)
    assert [each.datetime.isoformat() for each in built] == [
        "2025-06-01T00:00:00+00:00",
        "2025-06-11T00:00:00+00:00",
    ]
    assert all("start_datetime" not in each.properties for each in built)


def test_split_bands_become_one_asset_per_variable(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest", by_band=True)

    built = table.to_item(stac.create_item(paths))

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert sorted(built[0].assets) == ["nir", "red"]
    assert built[1].assets["nir"].href == str(paths[3])


def test_items_take_the_id_of_their_collection_without_a_link(scene, tmp_path):
    forest = stac.create_collection("forest", description="Forest samples")

    built = table.to_item(
        stac.create_item(_saved(scene, tmp_path, "forest"), collection=forest)
    )

    assert [each.collection_id for each in built] == ["forest", "forest"]
    assert [each.links for each in built] == [[], []]


def test_two_files_claiming_one_asset_key_refuse(scene, tmp_path):
    (first,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/a_{{time:%Y%m%d}}.tif")
    (second,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/b_{{time:%Y%m%d}}.tif")

    with pytest.raises(ValueError, match="both be the asset 'image'"):
        stac.create_item([first, second])


def test_an_item_states_what_the_file_stores(tmp_path):
    source = build_raster(times=2)[["red"]]
    path = tmp_path / "small.tif"
    # A GeoTIFF written by another tool stores the type it was told to.
    source.red.isel(time=0).rio.to_raster(path, dtype="uint8")

    (built,) = table.to_item(
        stac.create_item(
            {"red": path}, id="small", datetime=datetime(2025, 6, 1, tzinfo=UTC)
        )
    )

    assert source.red.dtype == "uint16"
    assert RasterExtension.ext(built.assets["red"]).bands[0].data_type == "uint8"


def test_no_paths_refuse():
    with pytest.raises(ValueError, match="saved raster files"):
        stac.create_item([])


def test_an_id_template_is_filled_per_scene(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = table.to_item(stac.create_item(paths, id="{lat:.2f}N_{start:%Y%m%d}"))

    assert [each.id for each in built] == ["45.13N_20250601", "45.13N_20250611"]
    assert built[0].assets["image"].href == str(paths[0])


def test_an_id_template_that_repeats_refuses(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    with pytest.raises(ValueError, match="same id"):
        stac.create_item(paths, id="{lat:.2f}N")


def test_building_items_computes_no_pixels(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    with dask.config.set(scheduler=_refuse):
        built = stac.create_item(paths)

    assert len(built) == 2


def test_a_timeless_raster_needs_a_time(label, tmp_path):
    files = _saved(label, tmp_path, "label")

    with pytest.raises(ValueError, match="pass datetime="):
        stac.create_item(files)

    (built,) = table.to_item(
        stac.create_item(files, datetime=datetime(2025, 6, 1, tzinfo=UTC))
    )
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_the_caller_names_the_assets_of_one_item(scene, label, tmp_path):
    (optical, _) = (str(file) for file in _saved(scene, tmp_path, "optical"))
    (classes,) = (str(file) for file in _saved(label, tmp_path, "label"))

    (built,) = table.to_item(
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
    (path,) = (str(file) for file in _saved(classes, tmp_path, "label"))

    (built,) = table.to_item(
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
    (path, _) = (str(file) for file in _saved(scene, tmp_path, "scene"))

    (built,) = table.to_item(stac.create_item({"image": path}, id="scene"))

    assert built.stac_extensions == [
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        EOExtension.get_schema_uri(),
    ]


def test_a_collection_starts_with_an_open_extent() -> None:
    forest = stac.create_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )

    assert (forest.id, forest.description, forest.license) == (
        "forest",
        "Forest samples",
        "CC-BY-4.0",
    )
    assert forest.extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]
    assert forest.extent.temporal.intervals == [[None, None]]


def test_a_collection_takes_stacs_own_fields() -> None:
    host = pystac.Provider("GeoSave", roles=[pystac.ProviderRole.HOST])

    forest = stac.create_collection(
        "forest",
        description="Forest samples",
        title="Forest",
        keywords=["forest", "label"],
        providers=[host],
    )

    assert (forest.title, forest.keywords) == ("Forest", ["forest", "label"])
    assert [provider.name for provider in forest.providers] == ["GeoSave"]
    assert forest.license == "other"


def test_every_extension_module_writes_onto_each_asset(scene, tmp_path) -> None:
    (path, _) = (str(file) for file in _saved(scene, tmp_path, "scene"))
    (built,) = table.to_item(stac.create_item({"image": path}, id="scene"))

    asset = built.assets["image"]
    assert ProjectionExtension.ext(asset).code == "EPSG:32633"
    assert len(RasterExtension.ext(asset).bands) == 2
    assert asset.media_type == "image/tiff; application=geotiff"
    assert asset.roles == ["data"]


def test_odc_stac_names_a_multiband_asset_after_its_bands(tmp_path: Path) -> None:
    import odc.stac

    scene = build_raster()
    (built,) = table.to_item(
        stac.create_item(
            _saved(scene, tmp_path, "forest"),
            datetime=datetime(2025, 6, 1, tzinfo=UTC),
        )
    )

    loaded = odc.stac.load([built], geobox=scene.gs.geobox, chunks={})

    assert list(loaded.data_vars) == ["red", "nir"]


def test_create_item_returns_one_row(label, tmp_path: Path) -> None:
    (path,) = (str(file) for file in _saved(label, tmp_path, "label"))

    rows = stac.create_item(
        {"label": path}, id="s0", datetime=datetime(2025, 6, 1, tzinfo=UTC)
    )

    assert rows["id"].tolist() == ["s0"]
    assert list(rows["assets"].iloc[0]) == ["label"]


def test_records_become_one_item_per_scene(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")

    rows = stac.create_item(files)

    assert len(rows) == 2
    item = table.to_item(rows)[0]
    assert sorted(item.assets) == ["nir", "red"]
    assert [b.name for b in EOExtension.ext(item.assets["red"]).bands] == ["red"]


def test_a_multiband_file_is_one_image_asset_with_bands_in_order(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    item = table.to_item(stac.create_item(files))[0]

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

    rows = stac.create_item(files, id="s0")

    assert rows["collection"].tolist() == ["optical", "optical", "label"]
    assert rows["id"].tolist()[2] == "s0/label"
    assert str(rows.iloc[2]["start_datetime"])[:10] == "2025-06-01"


def test_collection_overrides_what_files_state(scene, tmp_path) -> None:
    from geosave.geodata import attrs

    named = scene.gs.write_acdd(attrs.ACDD(id="sentinel-2-l2a"))
    paths = named.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    rows = stac.create_item(paths, collection="image")

    assert rows["collection"].tolist() == ["image", "image"]


def test_create_item_reads_each_file_with_its_storage_options(
    scene, tmp_path, monkeypatch
) -> None:
    from geosave.geodata.stac import create as module

    (path, _) = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    seen = []
    original = module.open_raster

    def spy(source, **options):
        seen.append(options.get("storage_options"))
        return original(source)

    monkeypatch.setattr(module, "open_raster", spy)

    stac.create_item([path], storage_options={"anon": True})

    assert seen == [{"anon": True}]

def test_saved_paths_become_one_item_each_named_after_its_file(label, tmp_path):
    north = label.expand_dims(time=[np.datetime64("2025-06-01")])
    # A label 10 km south lies on another grid, so it is another Item.
    south = north.assign_coords(y=north.y - 10_000)
    paths = []
    for name, tile in (("north", north), ("south", south)):
        (file,) = tile.gs.to_cog(f"{tmp_path}/{name}.tif")
        paths.append(Path(file))

    rows = stac.create_item(paths, collection="label")

    assert rows["id"].tolist() == ["north", "south"]
    assert rows["collection"].tolist() == ["label", "label"]
    assert [sorted(item.assets) for item in table.to_item(rows)] == [
        ["label"],
        ["label"],
    ]


def test_a_collection_can_be_named_by_its_id(scene, tmp_path):
    (path,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    one = stac.create_item({"image": str(path)}, id="s0", collection="image")
    many = stac.create_item([path], collection="image")

    assert one["collection"].tolist() == ["image"]
    assert many["collection"].tolist() == ["image"]


def test_named_assets_take_their_files_stem_without_an_id(label, tmp_path) -> None:
    (path,) = (str(file) for file in _saved(label, tmp_path, "label"))

    rows = stac.create_item({"label": path}, datetime=datetime(2025, 1, 1, tzinfo=UTC))

    assert rows["id"].tolist() == ["label"]


def test_records_may_arrive_as_a_generator(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    rows = stac.create_item(file for file in files)

    assert rows["id"].tolist() == ["forest_20250601", "forest_20250611"]


def test_an_id_without_fields_prefixes_several_scenes(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    rows = stac.create_item(files, id="s0")

    assert rows["id"].tolist() == ["s0/s_20250601", "s0/s_20250611"]


def test_an_id_without_fields_names_one_scene(scene, tmp_path) -> None:
    files = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    assert stac.create_item(files, id="s0")["id"].tolist() == ["s0"]


def test_an_id_template_fills_whatever_collections_files_state(scene, tmp_path):
    from geosave.geodata import attrs

    paths = []
    for day, collection in ((0, "sentinel-2-l2a"), (1, "landsat")):
        named = scene.isel(time=[day]).gs.write_acdd(attrs.ACDD(id=collection))
        paths += named.gs.to_cog(f"{tmp_path}/{collection}_{{time:%Y%m%d}}.tif")

    rows = stac.create_item(paths, id="{start:%Y%m%d}")

    assert rows["id"].tolist() == ["20250601", "20250611"]
