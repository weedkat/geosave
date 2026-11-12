"""Selected STAC Items load independently of discovery."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox

from geosave.geodata import io, stac
from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items

from tests.conftest import write_cogs
from datetime import datetime as dt
from datetime import timezone
import pystac
import geosave.geodata  # noqa: F401  — registers the .gs accessor
from geosave.geodata import open_raster
from geosave.geodata.attrs import CFPacking, EOBand
from geosave.geodata.features import ndvi
from geosave.geodata.stac.load import load_dataset
from geosave.geodata.stac import StacQuery


@pytest.fixture
def observation(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    raster = build_dataset(
        {"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))}, grid
    )
    raster = raster.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(raster, tmp_path / "red.tif")
    item = next(create_file_items({"red": path}))
    item.collection_id = "local"
    item.properties["eo:cloud_cover"] = 20.0
    return item, raster


def test_properties_follow_fused_time_and_keep_common_attrs(observation):
    item, raster = observation
    other = item.clone()
    other.id = "second"
    other.properties["eo:cloud_cover"] = 40.0
    item.properties["platform"] = other.properties["platform"] = "test"
    loaded = stac.load_dataset(
        [item, other],
        geobox=raster.gs.geobox,
        groupby="time",
        with_properties=["eo:cloud_cover"],
    )
    assert loaded.gs.stac_properties["eo:cloud_cover"].tolist() == [30.0]
    assert loaded.gs.stac_common_properties == {
        "platform": "test",
        "eo:cloud_cover": 30.0,
    }
    assert "eo:cloud_cover" not in loaded.data_vars


def test_loader_rejects_mixed_collections_before_reading(observation):
    item, raster = observation
    other = item.clone()
    other.collection_id = "other"
    with pytest.raises(ValueError, match="one collection"):
        stac.load_dataset([item, other], geobox=raster.gs.geobox)


def _add_coarse_asset(item, raster, tmp_path):
    """Give the Item a second asset on a grid twice as coarse."""
    coarse = build_dataset(
        {"swir": (("y", "x"), np.full((2, 2), 7, dtype="uint16"))},
        raster.gs.geobox.zoom_out(2),
    )
    coarse = coarse.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(coarse, tmp_path / "swir.tif")
    for key, asset in next(create_file_items({"swir": path})).assets.items():
        item.add_asset(key, asset)
    return coarse


def test_each_item_asset_is_one_node_on_its_stated_grid(observation, tmp_path):
    item, raster = observation
    coarse = _add_coarse_asset(item, raster, tmp_path)

    tree = stac.load_datatree([item])

    assert list(tree.gs.datasets) == [f"local/{item.id}/red", f"local/{item.id}/swir"]
    assert tree[f"local/{item.id}/red"].gs.geobox == raster.gs.geobox
    assert tree[f"local/{item.id}/swir"].gs.geobox == coarse.gs.geobox
    np.testing.assert_array_equal(
        tree[f"local/{item.id}/swir"].swir.values, coarse.swir.values
    )


def test_the_item_node_states_its_time_once_and_leaves_inherit_it(
    observation, tmp_path
):
    item, raster = observation
    _add_coarse_asset(item, raster, tmp_path)

    tree = stac.load_datatree([item])

    own = tree[f"local/{item.id}"].to_dataset(inherit=False)
    assert own.time.values == np.datetime64("2025-01-15")
    assert "stac_properties" in own.coords
    for leaf in tree[f"local/{item.id}"].leaves:
        assert "time" not in leaf.to_dataset(inherit=False).coords
        inherited = leaf.to_dataset(inherit="all_coords")
        assert inherited.time.values == np.datetime64("2025-01-15")


def test_the_item_node_states_its_period(observation):
    june, july = _months(observation)

    tree = stac.load_datatree([june, july])

    node = tree["local/july"].to_dataset(inherit=False)
    assert node.start_datetime.values == np.datetime64("2025-07-01")
    assert node.end_datetime.values == np.datetime64("2025-08-01")


def test_an_item_id_with_a_slash_cannot_name_a_node(observation):
    item, _ = observation
    item.id = "tile/20250115"

    with pytest.raises(ValueError, match="tile/20250115.*/"):
        stac.load_datatree([item])


def test_a_loaded_tree_round_trips_through_zarr(observation, tmp_path):
    from geosave.geodata import open_raster

    item, raster = observation
    _add_coarse_asset(item, raster, tmp_path)
    tree = stac.load_datatree([item]).compute()

    store = tree.gs.to_zarr(tmp_path / "tree.zarr")

    with open_raster(store) as restored:
        assert restored.isomorphic(tree)
        for leaf in tree.leaves:
            back = restored[leaf.path].to_dataset(inherit="all_coords")
            xr.testing.assert_equal(back, leaf.to_dataset(inherit="all_coords"))


def test_items_on_different_grids_each_keep_their_own(observation, tmp_path):
    item, raster = observation
    east = GeoBox.from_bbox(
        BoundingBox(500040, 5000000, 500080, 5000040, crs="EPSG:32633"),
        resolution=10,
    )
    shifted = build_dataset(
        {"red": (("y", "x"), np.ones((4, 4), "uint16"))}, east
    ).assign_coords(time=np.datetime64("2025-01-16"))
    other = next(
        create_file_items(
            {"red": io.formats.geotiff.write_cog(shifted, tmp_path / "east.tif")}
        )
    )
    other.collection_id = item.collection_id

    tree = stac.load_datatree([item, other])

    assert tree[f"local/{item.id}/red"].gs.geobox == raster.gs.geobox
    assert tree[f"local/{other.id}/red"].gs.geobox == east


def test_collections_are_sibling_nodes_stating_their_collection(observation):
    import pystac

    from geosave.geodata import attrs

    item, raster = observation
    other = item.clone()
    other.collection_id = "dem"
    collection = pystac.Collection(
        "local",
        "Local reflectance",
        pystac.Extent(
            pystac.SpatialExtent([[0, 0, 1, 1]]), pystac.TemporalExtent([[None, None]])
        ),
    )

    tree = stac.load_datatree([item, other], collections=[collection])

    assert list(tree.children) == ["local", "dem"]
    stated = attrs.StacCollection.from_attrs(tree["local"].attrs)
    assert stated.collection["id"] == "local"


def test_items_naming_no_collection_sit_under_default(observation):
    item, _ = observation
    item.collection_id = None

    tree = stac.load_datatree([item])

    assert list(tree.gs.datasets) == [f"default/{item.id}/red"]


def test_loading_a_tree_reads_no_pixels(observation, tmp_path):
    import dask

    from tests.geodata.conftest import _refuse

    item, raster = observation
    _add_coarse_asset(item, raster, tmp_path)

    with dask.config.set(scheduler=_refuse):
        tree = stac.load_datatree([item])

    for path in tree.gs.datasets:
        for variable in tree[path].data_vars.values():
            assert isinstance(variable.data, da.Array)


def test_signing_happens_at_load_without_mutating_items(observation):
    item, raster = observation
    original = item.assets["red"].href
    item.assets["red"].href = "https://unsigned.test/red.tif"
    urls = []

    def sign(url):
        urls.append(url)
        return original

    loaded = stac.load_dataset([item], geobox=raster.gs.geobox, patch_url=sign)
    np.testing.assert_array_equal(loaded.red.values[0], raster.red.values)
    assert urls == ["https://unsigned.test/red.tif"]
    assert item.assets["red"].href == "https://unsigned.test/red.tif"


def test_same_date_items_mosaic_by_default_and_stay_apart_by_id(observation):
    entry, raster = observation
    other = entry.clone()
    other.id = "second"
    mosaic = stac.load_dataset([entry, other], geobox=raster.gs.geobox)
    assert mosaic.sizes["time"] == 1
    separate = stac.load_dataset([entry, other], geobox=raster.gs.geobox, groupby="id")
    assert separate.sizes["time"] == 2


def _period_item(observation, ident, start, end):
    """Clone the observation as a composite Item covering [start, end)."""
    entry, _ = observation
    period = entry.clone()
    period.id = ident
    period.datetime = None
    period.properties["datetime"] = None
    period.properties.update(start_datetime=start, end_datetime=end)
    return period


def _months(observation):
    june = _period_item(
        observation, "june", "2025-06-01T00:00:00Z", "2025-07-01T00:00:00Z"
    )
    july = _period_item(
        observation, "july", "2025-07-01T00:00:00Z", "2025-08-01T00:00:00Z"
    )
    return june, july


def _days(values):
    return np.array(values, dtype="datetime64[ns]")


def test_composite_items_keep_their_own_periods_in_any_order(observation):
    _, raster = observation
    june, july = _months(observation)

    loaded = stac.load_dataset([july, june], geobox=raster.gs.geobox, groupby="id")

    for label, start in zip(
        loaded.time.values, loaded.start_datetime.values, strict=True
    ):
        assert start == label  # ODC labels a composite at its start
    ends = {
        str(start)[:10]: str(end)[:10]
        for start, end in zip(
            loaded.start_datetime.values, loaded.end_datetime.values, strict=True
        )
    }
    assert ends == {"2025-06-01": "2025-07-01", "2025-07-01": "2025-08-01"}


def test_scene_labels_state_no_period(observation):
    entry, raster = observation
    loaded = stac.load_dataset([entry], geobox=raster.gs.geobox, groupby="id")
    assert "start_datetime" not in loaded.coords


def test_a_solar_day_label_covers_its_local_day(tmp_path):
    from odc.geo.geom import BoundingBox as Box

    # Jakarta, UTC+7: 03:12 and 03:13 UTC share local Jan 10; 18:00 UTC is Jan 11.
    grid = GeoBox.from_bbox(
        Box(106.8, -6.3, 106.81, -6.29, crs="EPSG:4326"), resolution=0.0025
    )
    items = []
    for name, when in (
        ("a", "2025-01-10T03:12"),
        ("b", "2025-01-10T03:13"),
        ("c", "2025-01-10T18:00"),
    ):
        raster = build_dataset(
            {"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid
        ).assign_coords(time=np.datetime64(when))
        path = io.formats.geotiff.write_cog(raster, tmp_path / f"{name}.tif")
        entry = next(create_file_items({"red": path}))
        entry.id = name
        items.append(entry)

    loaded = stac.load_dataset(items, grid, groupby="solar_day")

    np.testing.assert_array_equal(
        loaded.start_datetime.values,
        _days(["2025-01-09T17:00", "2025-01-10T17:00"]),
    )
    np.testing.assert_array_equal(
        loaded.end_datetime.values,
        _days(["2025-01-10T17:00", "2025-01-11T17:00"]),
    )


def test_a_composite_keeps_its_own_period_under_solar_day(observation):
    _, raster = observation
    june, _ = _months(observation)

    loaded = stac.load_dataset([june], geobox=raster.gs.geobox, groupby="solar_day")

    np.testing.assert_array_equal(loaded.end_datetime.values, _days(["2025-07-01"]))


def test_a_multi_band_asset_holds_one_variable_per_band(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    pixels = {}
    for value, name in enumerate(["red", "green", "blue"]):
        pixels[name] = (("y", "x"), np.full(grid.shape, value, "uint16"))
    rgb = build_dataset(pixels, grid).assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(rgb, tmp_path / "visual.tif")
    item = next(create_file_items({"visual": path}))
    item.collection_id = "local"

    tree = stac.load_datatree([item])

    node = tree[f"local/{item.id}/visual"]
    assert list(node.data_vars) == ["red", "green", "blue"]
    np.testing.assert_array_equal(node.blue.values, 2)


def test_a_catalog_loads_one_collection_onto_a_chosen_grid(observation):
    from geosave.geodata.stac.catalog import create_catalog

    item, raster = observation
    grid = raster.gs.geobox.zoom_out(2)

    loaded = create_catalog([item]).gs.to_dataset(grid)

    assert loaded.gs.geobox == grid
    assert isinstance(loaded.red.data, da.Array)


def test_a_catalog_of_two_collections_does_not_load_as_one_dataset(observation):
    from geosave.geodata.stac.catalog import create_catalog

    item, raster = observation
    other = item.clone()
    other.id = "dem"
    other.collection_id = "dem"

    with pytest.raises(ValueError, match="one collection"):
        create_catalog([item, other]).gs.to_dataset(raster.gs.geobox)


def test_items_sharing_a_grid_load_together_and_keep_their_own_facts(observation):
    from geosave.geodata.stac.raster import read_properties_coordinate

    item, raster = observation
    later = item.clone()
    later.id = "later"
    later.datetime = item.datetime.replace(day=20)
    later.properties["eo:cloud_cover"] = 40.0
    tile = item.clone()
    tile.id = "tile"
    tile.properties["eo:cloud_cover"] = 60.0

    tree = stac.load_datatree([item, later, tile])

    covers = {}
    for entry in (item, later, tile):
        node = tree[f"local/{entry.id}"].to_dataset(inherit=False)
        covers[entry.id] = read_properties_coordinate(node)[0]["eo:cloud_cover"]
        leaf = tree[f"local/{entry.id}/red"].to_dataset(inherit="all_coords")
        assert leaf.time.values == np.datetime64(
            entry.datetime.replace(tzinfo=None), "ns"
        )
        np.testing.assert_array_equal(leaf.red.values, raster.red.values)
    assert covers == {item.id: 20.0, "later": 40.0, "tile": 60.0}


@pytest.mark.parametrize("fill", [-9999, 0.1])
def test_effective_nodata_follows_loaded_pixels(local_catalog, fill):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    loaded = load_dataset(
        matched,
        geobox=grid,
        collection=collection,
        **{**options, "dtype": "float32", "nodata": fill},
    )

    assert isinstance(loaded.red.data, da.Array)
    assert loaded.red.values[0, 0, 0] == np.float32(fill)
    assert loaded.red.attrs["_FillValue"] == np.float32(fill)
    prepared = loaded.gs.mask_and_scale()
    assert np.isnan(prepared.red.values[0, 0, 0])
    assert prepared.red.values[0, 1, 1] == pytest.approx(0.2)


@pytest.mark.parametrize("lazy", [False, True])
def test_ingestion_decode_matches_deferred_cf_and_roundtrips(
    local_catalog, tmp_path, lazy
):
    from dask.callbacks import Callback

    client, region, grid, options = local_catalog
    items = client.search_items(StacQuery(collections=["local"]).set_region(region))
    options = {
        **options,
        "chunks": {"x": 1, "y": 1} if lazy else None,
        "dtype": "float32",
        "nodata": 0.1,
    }
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        raw = load_dataset(items, grid, **options)
        physical = load_dataset(items, grid, mask_and_scale=True, **options)
        deferred = raw.gs.mask_and_scale()
    if lazy:
        assert not tasks
        assert isinstance(physical.red.data, da.Array)
    xr.testing.assert_identical(physical, deferred)
    xr.testing.assert_identical(physical, xr.decode_cf(raw, decode_coords="all"))
    xr.testing.assert_identical(physical.coords.to_dataset(), raw.coords.to_dataset())
    assert physical.gs.geobox == grid
    assert np.isnan(physical.red.values[0, 0, 0])
    assert physical.red.values[0, 1, 1] == pytest.approx(0.2)
    assert physical.nir.values[0, 1, 1] == pytest.approx(1.1)

    path = tmp_path / "decoded.nc"
    physical.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as reopened:
        xr.testing.assert_allclose(reopened, physical)
        assert reopened.gs.geobox == grid
    with xr.open_dataset(path, decode_coords="all", mask_and_scale=False) as stored:
        np.testing.assert_array_equal(stored.red, raw.red)
        assert stored.red.dtype == raw.red.dtype


def test_aliases_resolve_each_multiband_asset_index(local_catalog):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    loaded = load_dataset(
        matched, geobox=grid, collection=collection, **{**options, "chunks": None}
    )

    assert isinstance(loaded.red.data, np.ndarray)
    assert loaded.red.dims == ("time", "y", "x")
    assert loaded.sizes["time"] == 1
    assert loaded.red.attrs["scale_factor"] == 0.0001
    assert loaded.nir.attrs["scale_factor"] == 0.0002
    assert loaded.nir.attrs["add_offset"] == -0.1
    assert loaded.gs.geobox == grid
    assert loaded.gs.mask_and_scale().nir.values[0, 1, 1] == pytest.approx(1.1)


def test_a_load_carries_its_collection_and_the_properties_every_item_shares(
    local_catalog,
) -> None:

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    first = matched[0]
    first.properties.update(platform="sentinel-2a", constellation="sentinel-2")
    second = first.clone()
    second.id = "second"
    second.properties["platform"] = "sentinel-2b"
    matched = [first, second]

    loaded = load_dataset(matched, geobox=grid, collection=collection, **options)

    assert loaded.gs.stac_collection.id == collection.id
    assert loaded.gs.stac_collection.license == collection.license
    assert loaded.gs.stac_common_properties == {"constellation": "sentinel-2"}
    assert loaded.gs.stac_properties["constellation"].tolist() == ["sentinel-2"]
    assert "title" not in loaded.attrs


@pytest.mark.parametrize("groupby", ["id", "time"])
@pytest.mark.parametrize(
    "properties",
    [
        None,
        ["eo:cloud_cover"],
        [
            {"key": "start_datetime", "dtype": "str"},
            {"key": "end_datetime", "dtype": "str"},
        ],
    ],
)
def test_loader_keeps_odc_time_and_property_representation(
    local_catalog, groupby, properties
):
    from copy import deepcopy

    import odc.stac

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    composite = matched[0]
    instant = composite.clone()
    instant.id = "instant"
    instant.datetime = composite.datetime
    instant.properties.pop("datetime", None)
    composite.properties["eo:cloud_cover"] = 20.0
    instant.properties["eo:cloud_cover"] = 40.0
    composite.properties.update(
        start_datetime="2025-05-31T00:00:00Z",
        end_datetime="2025-06-01T23:59:59Z",
    )
    matched = [composite, instant]
    original = deepcopy(instant.properties)

    configured = dict(
        groupby=groupby,
        with_properties=properties,
    )
    expected = odc.stac.load(
        [composite, instant],
        geobox=grid,
        **{
            **options,
            **configured,
            "with_properties": [
                dict(key=p, name=p) if isinstance(p, str) else {**p, "name": p["key"]}
                for p in properties or []
            ],
        },
    )
    loaded = load_dataset(
        matched, geobox=grid, collection=collection, **{**options, **configured}
    )

    keys = [
        request if isinstance(request, str) else request["key"]
        for request in properties or []
    ]
    assert isinstance(loaded.red.data, da.Array)
    assert dict(loaded.sizes) == dict(expected.sizes)
    assert set(loaded.data_vars) == set(expected.data_vars) - set(keys)
    np.testing.assert_array_equal(loaded.time.values, expected.time.values)
    np.testing.assert_array_equal(loaded.red.values, expected.red.values)
    # Each date's mapping holds odc-stac's own value of each requested key; its
    # NaN, or 'None' text, for an Item lacking the key leaves the key absent.
    recorded = loaded.gs.stac_properties
    for key in keys:
        published = np.atleast_1d(expected[key].values).tolist()
        assert _present(recorded[key].tolist()) == _present(published)
    assert ("stac_properties" in loaded.coords) == bool(keys)
    assert instant.properties == original
    if not properties:
        assert "start_datetime" not in loaded
        assert "end_datetime" not in loaded


def _present(values: list) -> list:
    """Spell every missing value None: odc-stac's NaN or 'None' text, or a table's NaN."""
    present = []
    for value in values:
        missing = value is None or value == "None"
        if isinstance(value, float) and np.isnan(value):
            missing = True
        present.append(None if missing else value)
    return present


def _apply_metadata(items, collection, loaded):
    from geosave.geodata.stac.load import stamp_band_attrs

    return stamp_band_attrs(items, loaded)


def _item(id: str, timestamp: dt, fields: dict[str, object]) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=timestamp.replace(tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            media_type=pystac.MediaType.GEOTIFF,
            extra_fields={"raster:bands": [fields]},
        ),
    )
    return item


def _collection() -> pystac.Collection:
    return pystac.Collection(
        id="example",
        description="Example collection",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[dt(2025, 1, 1, tzinfo=timezone.utc), None]]),
        ),
        license="CC-BY-4.0",
    )


@pytest.mark.parametrize(
    "field,value", [("scale", 0.0002), ("offset", 1.0), ("scale", None)]
)
def test_load_rejects_incompatible_source_conversion(local_catalog, field, value):
    client, region, grid, options = local_catalog
    first = client.search_items(StacQuery(collections=["local"]).set_region(region))[0]
    second = first.clone()
    second.id = "second"
    second.datetime = dt(2025, 1, 16, tzinfo=timezone.utc)
    fields = second.assets["reflectance"].extra_fields["raster:bands"][0]
    if value is None:
        fields.pop(field)
    else:
        fields[field] = value

    with pytest.raises(ValueError, match="red.*incompatible source scale/offset"):
        load_dataset(
            [first, second],
            geobox=grid,
            **{**options, "chunks": None},
        )

    # A conflict on red must not prevent loading the selected, uniform nir band.
    loaded = load_dataset(
        [first, second],
        geobox=grid,
        **{**options, "bands": "nir", "chunks": None},
    ).gs.mask_and_scale()
    np.testing.assert_allclose(loaded.nir.values[:, 1, 1], [1.1, 1.1])


def test_identity_source_conversion_accepts_omitted_fields(local_catalog):
    client, region, grid, options = local_catalog
    first = client.search_items(StacQuery(collections=["local"]).set_region(region))[0]
    first.assets["reflectance"].extra_fields["raster:bands"][0]["scale"] = 1.0
    second = first.clone()
    second.id = "second"
    second.datetime = dt(2025, 1, 16, tzinfo=timezone.utc)
    second.assets["reflectance"].extra_fields["raster:bands"][0].pop("scale")

    loaded = load_dataset(
        [first, second],
        geobox=grid,
        **{**options, "bands": "red", "chunks": None},
    ).gs.mask_and_scale()
    np.testing.assert_array_equal(loaded.red.values[:, 1, 1], [2000, 2000])


def test_an_item_without_the_asset_keeps_published_fields() -> None:
    first = _item("scene-1", dt(2025, 1, 1), {"unit": "1"})
    second = _item("scene-2", dt(2025, 1, 2), {"unit": "1"})
    del second.assets["red"]

    loaded = _apply_metadata(
        [first, second],
        _collection(),
        xr.Dataset({"red": (("y", "x"), [[1]])}),
    )

    assert loaded.red.attrs["units"] == "1"


def test_reading_writes_no_packing_when_no_item_publishes_it() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {"unit": "1"}),
    ]

    loaded = _apply_metadata(
        items, _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.gs.packing is None


def test_reading_keeps_a_label_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    loaded = _apply_metadata(
        items, _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.attrs["units"] == "1"


def test_loader_nodata_is_authoritative_over_source_values() -> None:
    items = [
        _item("first", dt(2025, 1, 1), {"nodata": 0}),
        _item("second", dt(2025, 1, 2), {"nodata": 1}),
    ]
    loaded = xr.Dataset(
        {"red": xr.DataArray([[1.0]], dims=("y", "x"), attrs={"nodata": -9999})}
    )

    loaded = _apply_metadata(items, _collection(), loaded)

    assert loaded.red.gs.nodata == -9999


def test_reader_uses_the_selected_band_without_mutating_the_item() -> None:
    item = _item("multi", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "raster:bands": [{"unit": "1", "scale": 0.1}, {"unit": "K", "offset": 0}],
        "eo:bands": [{"description": "Reflectance"}, {"description": "Temperature"}],
    }
    before = item.to_dict()

    loaded = _apply_metadata(
        [item],
        _collection(),
        xr.Dataset({"red.1": (("y", "x"), [[1]]), "red.2": (("y", "x"), [[2]])}),
    )

    assert loaded["red.1"].attrs == {
        "units": "1",
        "long_name": "Reflectance",
        "scale_factor": 0.1,
    }
    assert loaded["red.1"].gs.packing == CFPacking(scale_factor=0.1)
    assert loaded["red.2"].attrs == {"units": "K", "long_name": "Temperature"}
    assert item.to_dict() == before


def test_loader_packing_overrides_the_agreed_source_packing() -> None:
    item = _item("scene", dt(2025, 1, 1), {"scale": 0.1, "offset": 2})
    loaded = xr.Dataset(
        {"red": xr.DataArray([[1]], dims=("y", "x"), attrs={"scale_factor": 1.0})}
    )

    loaded = _apply_metadata([item], _collection(), loaded)

    assert loaded.red.gs.packing == CFPacking(scale_factor=1, add_offset=2)


def test_spectral_facts_reach_the_variable() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]
    }

    loaded = _apply_metadata(
        [item], _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.gs.spectral == EOBand(common_name="red", center_wavelength=0.665)


def test_observations_to_prepared_index_to_reusable_dataset(local_catalog, tmp_path):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    raw = load_dataset(matched, geobox=grid, collection=collection, **options)
    reflectance = raw.gs.mask_and_scale()
    derived = reflectance.assign(ndvi=ndvi(reflectance, nir="nir", red="red"))[["ndvi"]]
    composite = derived.median("time", keep_attrs=True)
    assert isinstance(composite.ndvi.data, da.Array)
    assert composite.gs.geobox == grid
    written = composite.gs.to_zarr(tmp_path / "ndvi.zarr")

    with open_raster(written, chunks={}) as reopened:
        assert isinstance(reopened.ndvi.data, da.Array)
        assert reopened.gs.geobox == grid
        assert np.isnan(reopened.ndvi.values[0, 0])
        assert reopened.ndvi.values[1, 1] == pytest.approx((1.1 - 0.2) / (1.1 + 0.2))
        assert "scale_factor" not in reopened.ndvi.attrs
    # Preparation leaves the reusable source configuration and packed input intact.
    assert raw.red.attrs["scale_factor"] == 0.0001
    assert raw.red.values[0, 1, 1] == 2000


def test_a_scene_states_the_period_its_item_states(local_catalog, tmp_path):

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    item = matched[0]
    item.properties.update(
        start_datetime="2025-01-01T00:00:00Z", end_datetime="2025-01-31T23:59:59Z"
    )
    matched = [item]
    loaded = load_dataset(matched, geobox=grid, collection=collection, **options)
    rebuilt = next(
        create_file_items(write_cogs(loaded, tmp_path, format_paths="composite.tif"))
    )
    assert str(loaded.start_datetime.values[0])[:10] == "2025-01-01"
    assert str(loaded.end_datetime.values[0])[:19] == "2025-01-31T23:59:59"
    assert rebuilt.properties["start_datetime"].startswith("2025-01-01")
    assert rebuilt.properties["end_datetime"].startswith("2025-01-31T23:59:59")


def test_loaded_stac_metadata_is_interpretable_without_pixel_reads(local_catalog):
    from dask.callbacks import Callback

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    with Callback(
        pretask=lambda *_: pytest.fail("metadata inspection computed pixels")
    ):
        loaded = load_dataset(matched, geobox=grid, collection=collection, **options)
        assert loaded.gs.spatial_dims == ("y", "x")
        assert loaded.gs.time_coordinate.name == "time"
        mappings = loaded.cf.grid_mappings
        assert len(mappings) == 1
        assert mappings[0].crs == grid.crs.proj
        assert loaded.red.gs.packing.scale_factor == 0.0001
        assert loaded.nir.gs.packing.scale_factor == 0.0002
        assert "nodata" not in loaded.red.attrs
        assert loaded.gs.stac_collection.id == collection.id
