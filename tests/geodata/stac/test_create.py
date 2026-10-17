"""Collections and Items built from the rasters a writer saved."""

from __future__ import annotations

from geosave.geodata.stac.translate import table as stac_table

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

from geosave.geodata import io, raster, stac, tree
from geosave.geodata import open_vector
from geosave.geodata.io.raster.file import RasterFile, Scene
from geosave.geodata.stac.translate.item import TIMELESS, read_timespan
from geosave.geodata.warnings import TimelessItemWarning

from tests.geodata.conftest import build_raster, cog_template, flat, tree_templates

IDS = ["forest_20250601T000000", "forest_20250602T000000"]


def _saved(data, root, name, *, by_band=False):
    """Write a raster or tree in the layout these Items are named after."""
    if isinstance(data, xr.DataTree):
        return data.gs.to_cog(tree_templates(data, root / name))
    template = cog_template(root, name, dated="time" in data.dims)
    if by_band:
        template = template.removesuffix(".tif") + "/{band}.tif"
    return data.gs.to_cog(template)


def _items(files, **options) -> list[pystac.Item]:
    """Build one Item per scene a writer returned."""
    id = options.pop("id", "forest_{start:%Y%m%dT%H%M%S}")
    return list(stac.create_items(files, id_format=id, **options))


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    source = build_raster(times=2)
    for built in _items(_saved(source, tmp_path, "forest")):
        built.validate()


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))}, grid
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    built = next(stac.create_items(Scene.open({"label": path}), id_format="laea"))

    assert built.bbox[0] == pytest.approx(56.78, abs=0.01)
    assert built.bbox[1] == pytest.approx(12.34, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    source = build_raster(times=2)
    rows = stac.create_catalog(_items(_saved(source, tmp_path, "forest", by_band=True)))

    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = stac_table.to_items(open_vector(saved))
    assert [entry.id for entry in restored] == IDS
    assert sorted(restored[0].assets) == ["nir", "red"]
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"


def test_one_file_per_scene_is_one_item_per_instant(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = _items(paths)

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert [list(each.assets) for each in built] == [["image"], ["image"]]
    assert built[1].assets["image"].href == str(paths[1].written["image"])
    assert built[1].datetime == datetime(2025, 6, 11, tzinfo=UTC)
    assert [each.datetime.isoformat() for each in built] == [
        "2025-06-01T00:00:00+00:00",
        "2025-06-11T00:00:00+00:00",
    ]
    assert all("start_datetime" not in each.properties for each in built)


def test_split_bands_become_one_asset_per_variable(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest", by_band=True)

    built = _items(paths)

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert sorted(built[0].assets) == ["nir", "red"]
    assert built[1].assets["nir"].href == str(paths[1].written["nir"])


def test_items_take_the_id_of_their_collection_without_a_link(scene, tmp_path):
    forest = stac.create_collection("forest", description="Forest samples")

    built = _items(_saved(scene, tmp_path, "forest"), collection=forest)

    assert [each.collection_id for each in built] == ["forest", "forest"]
    assert [each.links for each in built] == [[], []]


def test_an_item_states_what_the_file_stores(tmp_path):
    source = build_raster(times=2)[["red"]]
    path = tmp_path / "small.tif"
    # A GeoTIFF written by another tool stores the type it was told to.
    source.red.isel(time=0).rio.to_raster(path, dtype="uint8")

    built = next(
        stac.create_items(
            Scene.open({"red": path}),
            id_format="small",
            datetime=datetime(2025, 6, 1, tzinfo=UTC),
        )
    )

    assert source.red.dtype == "uint16"
    assert RasterExtension.ext(built.assets["red"]).bands[0].data_type == "uint8"


def test_an_id_template_is_filled_per_scene(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    built = _items(paths, id="{lat:.2f}N_{start:%Y%m%d}")

    assert [each.id for each in built] == ["45.13N_20250601", "45.13N_20250611"]
    assert built[0].assets["image"].href == str(paths[0].written["image"])


def test_layers_of_one_tree_share_an_id_across_collections(scene, tmp_path):
    built_tree = tree({"image": scene.isel(time=0), "label": scene.isel(time=0)})
    paths = built_tree.gs.to_cog(tmp_path / "{collection}/tile.tif")

    built = _items(paths, id="tile")

    assert [each.id for each in built] == ["tile", "tile"]
    assert [each.collection_id for each in built] == ["image", "label"]


def test_building_items_computes_no_pixels(scene, tmp_path):
    paths = _saved(scene, tmp_path, "forest")

    with dask.config.set(scheduler=_refuse):
        built = _items(paths)

    assert len(built) == 2


def test_a_timeless_raster_spans_all_of_time(label, tmp_path):
    (written,) = _saved(label, tmp_path, "label")

    with pytest.warns(TimelessItemWarning, match="pass datetime="):
        built = next(stac.create_items(written, id_format="label"))

    assert built.common_metadata.start_datetime == TIMELESS[0]
    assert built.common_metadata.end_datetime == TIMELESS[1]
    (restored,) = stac_table.to_items(stac.create_catalog([built]))
    assert read_timespan(restored) is None


def test_a_timeless_raster_takes_the_time_given(label, tmp_path):
    (written,) = _saved(label, tmp_path, "label")

    built = next(
        stac.create_items(
            written, id_format="label", datetime=datetime(2025, 6, 1, tzinfo=UTC)
        )
    )

    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_the_caller_names_the_assets_of_one_item(scene, label, tmp_path):
    (optical, _) = flat(_saved(scene, tmp_path, "optical"))
    (classes,) = flat(_saved(label, tmp_path, "label"))

    built = next(
        stac.create_items(
            Scene(None, {"optical": optical, "label": classes}), id_format="s0"
        )
    )

    assert built.id == "s0"
    assert sorted(built.assets) == ["label", "optical"]
    assert built.assets["label"].href == str(classes.path)
    # The timeless label rides on the instant its dated sibling covers.
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_a_store_is_an_item_of_one_asset(scene, tmp_path) -> None:
    store = scene.isel(time=0).gs.to_zarr(tmp_path / "cube.zarr")

    built = next(stac.create_items(store, id_format="cube"))

    assert list(built.assets) == ["image"]
    assert built.assets["image"].media_type == pystac.MediaType.ZARR
    assert built.assets["image"].href == str(store.written["image"])


def test_a_file_outside_any_raster_format_refuses(scene, tmp_path) -> None:
    (file,) = flat(scene.isel(time=0).gs.to_cog(f"{tmp_path}/scene.tif"))
    renamed = RasterFile(file.path.with_suffix(".bin"), file.header)

    with pytest.raises(ValueError, match="names no raster file"):
        next(stac.create_items(Scene(None, {"image": renamed}), id_format="scene"))


def test_a_pending_file_must_be_computed_first(scene, tmp_path) -> None:
    pending = scene.gs.to_zarr(tmp_path / "cube.zarr", compute=False)

    with pytest.raises(RuntimeError, match="compute"):
        next(stac.create_items(pending, id_format="cube"))


def test_a_label_item_declares_only_what_it_uses(label, tmp_path: Path) -> None:
    classes = label.gs.write_legend({1: "forest"}, target="label")
    (path,) = flat(_saved(classes, tmp_path, "label"))

    built = next(
        stac.create_items(
            Scene(None, {"label": path}),
            id_format="label",
            datetime=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )

    assert set(built.stac_extensions) == {
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        EOExtension.get_schema_uri(),
        ClassificationExtension.get_schema_uri(),
    }


def test_a_plain_scene_declares_its_grid_bands_and_time(scene, tmp_path: Path) -> None:
    (path, _) = flat(_saved(scene, tmp_path, "scene"))

    built = next(stac.create_items(Scene(None, {"image": path}), id_format="scene"))

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
    (path, _) = flat(_saved(scene, tmp_path, "scene"))
    built = next(stac.create_items(Scene(None, {"image": path}), id_format="scene"))

    asset = built.assets["image"]
    assert ProjectionExtension.ext(asset).code == "EPSG:32633"
    assert len(RasterExtension.ext(asset).bands) == 2
    assert asset.media_type == "image/tiff; application=geotiff"
    assert asset.roles == ["data"]


def test_odc_stac_names_a_multiband_asset_after_its_bands(tmp_path: Path) -> None:
    import odc.stac

    scene = build_raster()
    (built,) = _items(
        _saved(scene, tmp_path, "forest"),
        id="forest",
        datetime=datetime(2025, 6, 1, tzinfo=UTC),
    )

    loaded = odc.stac.load([built], geobox=scene.gs.geobox, chunks={})

    assert list(loaded.data_vars) == ["red", "nir"]


def test_records_become_one_item_per_scene(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")

    built = _items(files)

    assert len(built) == 2
    item = built[0]
    assert sorted(item.assets) == ["nir", "red"]
    assert [b.name for b in EOExtension.ext(item.assets["red"]).bands] == ["red"]


def test_a_multiband_file_is_one_image_asset_with_bands_in_order(
    scene, tmp_path
) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    item = next(stac.create_items(files[0], id_format="forest_20250601"))

    assert list(item.assets) == ["image"]
    names = [b.name for b in EOExtension.ext(item.assets["image"]).bands]
    assert names == ["red", "nir"]


def test_a_trees_layers_become_collections(scene, label, tmp_path) -> None:
    files = tree({"optical": scene, "label": label}).gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    built = [
        next(
            stac.create_items(
                each, id_format="s0", datetime=datetime(2025, 6, 1, tzinfo=UTC)
            )
        )
        for each in files
    ]

    assert [each.collection_id for each in built] == ["optical", "optical", "label"]
    assert built[2].datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_collection_overrides_what_files_state(scene, tmp_path) -> None:
    from geosave.geodata import attrs

    named = scene.gs.write_acdd(attrs.ACDD(id="sentinel-2-l2a"))
    paths = named.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    built = _items(paths, collection="image")

    assert [each.collection_id for each in built] == ["image", "image"]


def test_create_item_does_not_reopen_a_file_a_writer_returned(
    scene, tmp_path, monkeypatch
) -> None:
    from geosave.geodata.io import openers

    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    def refuse(*args, **options):
        raise AssertionError("a returned file was reopened")

    monkeypatch.setattr(openers, "open_raster", refuse)

    assert len(_items(files)) == 2


def test_a_file_is_keyed_by_its_band_or_image_where_it_holds_several(
    scene, label, tmp_path
):
    (one_band,) = label.gs.to_cog(f"{tmp_path}/a/label.tif")
    (several,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/b/scene.tif")
    (split,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/c/{{band}}.tif")

    assert list(one_band.written) == ["label"]
    assert list(several.written) == ["image"]
    assert list(split.written) == ["red", "nir"]


def test_a_collection_can_be_named_by_its_id(scene, tmp_path):
    (item,) = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    named = next(stac.create_items(item, id_format="s0", collection="image"))
    given = next(
        stac.create_items(
            item,
            id_format="s0",
            collection=stac.create_collection("image", description="Images"),
        )
    )

    assert named.collection_id == given.collection_id == "image"


def test_an_id_without_fields_names_the_item(scene, tmp_path) -> None:
    files = scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    (file,) = files

    assert next(stac.create_items(file, id_format="s0")).id == "s0"


def test_an_id_template_fills_whatever_collections_files_state(scene, tmp_path):
    from geosave.geodata import attrs

    paths = []
    for day, collection in ((0, "sentinel-2-l2a"), (1, "landsat")):
        named = scene.isel(time=[day]).gs.write_acdd(attrs.ACDD(id=collection))
        paths += named.gs.to_cog(f"{tmp_path}/{collection}_{{time:%Y%m%d}}.tif")

    built = _items(paths, id="{start:%Y%m%d}")

    assert [each.id for each in built] == ["20250601", "20250611"]


def test_create_items_describes_saved_paths_without_computing_pixels(tmp_path):
    from dask.callbacks import Callback

    source = build_raster(times=1).isel(time=0)
    path = tmp_path / "image.tif"
    io.geotiff.write_cog(source, path)

    collection = stac.create_collection("optical", description="Test imagery")
    with Callback(start=_refuse):
        built = next(
            stac.create_items({"image": path}, collection=collection, id_format="tile")
        )

    assert built.id == "tile"
    assert built.collection_id == "optical"
    assert list(built.assets) == ["image"]
    assert built.assets["image"].href == str(path)
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    assert ProjectionExtension.ext(built.assets["image"]).shape == [2, 2]
    restored = open_vector(
        stac.create_catalog([built], collections=[collection]).gs.to_geoparquet(
            tmp_path / "catalog.parquet"
        )
    )
    opened = restored.iloc[0].gs.to_xarray()
    assert opened.gs.geobox == source.gs.geobox
    xr.testing.assert_equal(
        opened.drop_vars("spatial_ref"), source.drop_vars("spatial_ref")
    )
