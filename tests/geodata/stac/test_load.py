"""Selected STAC Items load independently of discovery."""

from __future__ import annotations

from datetime import datetime as dt
from datetime import timezone

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox

from geosave.geodata import io, open_raster, stac
from geosave.geodata.features import ndvi
from geosave.geodata.stac import StacQuery
from geosave.geodata.stac.load import group_bands_by_grid, load_dataset
from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items


@pytest.fixture
def observation(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    raster = build_dataset(
        {"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))}, grid
    )
    raster = raster.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.raster.geotiff.write_cog(raster, tmp_path / "red.tif")
    item = next(create_file_items({"red": path}))
    item.collection_id = "local"
    item.properties["eo:cloud_cover"] = 20.0
    return item, raster


def _add_coarse_asset(item, raster, tmp_path):
    """Give the Item a second asset on a grid twice as coarse."""
    coarse = build_dataset(
        {"swir": (("y", "x"), np.full((2, 2), 7, dtype="uint16"))},
        raster.gs.geobox.zoom_out(2),
    )
    coarse = coarse.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.raster.geotiff.write_cog(coarse, tmp_path / "swir.tif")
    for key, asset in next(create_file_items({"swir": path})).assets.items():
        item.add_asset(key, asset)
    return coarse


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
        chunks={},
        with_properties=["eo:cloud_cover", {"key": "platform", "dtype": "str"}],
    )
    # ODC's default fuser averages numbers and joins text of a fused step;
    # the mark keeps the STAC key.
    assert loaded["eo_cloud_cover"].values.tolist() == [30.0]
    assert loaded["platform"].values.tolist() == ["test,test"]
    assert not loaded.data_vars.keys() & {"eo_cloud_cover", "platform"}
    assert loaded.gs.stac_properties == {
        "eo:cloud_cover": "eo_cloud_cover",
        "platform": "platform",
    }


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
    # ODC also patches its virtual property sources; only the asset is signed.
    signed = []
    for url in urls:
        if not url.startswith("virtual://"):
            signed.append(url)
    assert signed == ["https://unsigned.test/red.tif"]
    assert item.assets["red"].href == "https://unsigned.test/red.tif"


def test_same_date_items_mosaic_by_default_and_stay_apart_by_id(observation):
    entry, raster = observation
    other = entry.clone()
    other.id = "second"
    mosaic = stac.load_dataset([entry, other], geobox=raster.gs.geobox)
    assert mosaic.sizes["time"] == 1
    separate = stac.load_dataset([entry, other], geobox=raster.gs.geobox, groupby="id")
    assert separate.sizes["time"] == 2


@pytest.mark.parametrize("fill", [-9999])
def test_effective_nodata_follows_loaded_pixels(local_catalog, fill):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    loaded = load_dataset(
        matched,
        geobox=grid,
        **{**options, "dtype": "float32", "nodata": fill},
    )

    assert isinstance(loaded.red.data, da.Array)
    assert loaded.red.values[0, 0, 0] == np.float32(fill)
    assert loaded.red.attrs["_FillValue"] == np.float32(fill)
    prepared = loaded.gs.mask_and_scale()
    assert np.isnan(prepared.red.values[0, 0, 0])
    assert prepared.red.values[0, 1, 1] == pytest.approx(0.2)


def test_ingestion_decode_matches_deferred_cf_and_roundtrips(local_catalog, tmp_path):
    lazy = True
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
        raw = load_dataset(items, geobox=grid, **options)
        physical = raw.gs.mask_and_scale()
    if lazy:
        assert not tasks
        assert isinstance(physical.red.data, da.Array)
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
    loaded = load_dataset(matched, geobox=grid, **{**options, "chunks": None})

    assert isinstance(loaded.red.data, np.ndarray)
    assert loaded.red.dims == ("time", "y", "x")
    assert loaded.sizes["time"] == 1
    assert loaded.red.attrs["scale_factor"] == 0.0001
    assert loaded.nir.attrs["scale_factor"] == 0.0002
    assert loaded.nir.attrs["add_offset"] == -0.1
    assert loaded.gs.geobox == grid
    assert loaded.gs.mask_and_scale().nir.values[0, 1, 1] == pytest.approx(1.1)


def test_a_load_carries_its_collection_and_requested_properties(
    local_catalog,
) -> None:

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    first = matched[0]
    first.properties.update(platform="sentinel-2a", constellation="sentinel-2")
    second = first.clone()
    second.id = "second"
    second.properties["platform"] = "sentinel-2b"
    matched = [first, second]

    loaded = load_dataset(
        matched,
        geobox=grid,
        with_properties=[{"key": "constellation", "dtype": "str"}],
        **options,
    )

    assert loaded.gs.stac_collection.id == client.collection("local").id
    assert loaded.gs.stac_collection.license == client.collection("local").license
    # Only requested properties load, one value per fused step.
    assert loaded["constellation"].values.tolist() == ["sentinel-2,sentinel-2"]
    assert "platform" not in loaded.coords
    assert "title" not in loaded.attrs


@pytest.mark.parametrize("field,value", [("scale", 0.0002)])
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


def test_observations_to_prepared_index_to_reusable_dataset(local_catalog, tmp_path):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    raw = load_dataset(matched, geobox=grid, **options)
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


def test_loaded_stac_metadata_is_interpretable_without_pixel_reads(local_catalog):
    from dask.callbacks import Callback

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    with Callback(
        pretask=lambda *_: pytest.fail("metadata inspection computed pixels")
    ):
        loaded = load_dataset(matched, geobox=grid, **options)
        assert loaded.gs.spatial_dims == ("y", "x")
        assert loaded.gs.time_coordinate.name == "time"
        mappings = loaded.cf.grid_mappings
        assert len(mappings) == 1
        assert mappings[0].crs == grid.crs.proj
        assert loaded.red.gs.packing.scale_factor == 0.0001
        assert loaded.nir.gs.packing.scale_factor == 0.0002
        assert "nodata" not in loaded.red.attrs
        assert loaded.gs.stac_collection.id == client.collection("local").id


def test_like_copies_a_raster_grid(observation):
    item, raster = observation

    loaded = stac.load_dataset([item], like=raster, chunks={})

    assert loaded.gs.geobox == raster.gs.geobox


def test_a_step_states_the_period_of_the_items_it_fuses(observation, tmp_path):
    item, raster = observation
    item.properties.update(
        start_datetime="2025-01-15T10:30:00Z", end_datetime="2025-01-15T10:30:40Z"
    )
    later = item.clone()
    later.id = "later"
    later.properties.update(
        start_datetime="2025-01-15T10:31:00+00:00",
        end_datetime="2025-01-15T10:31:40Z",
    )

    loaded = stac.load_dataset(
        [item, later], geobox=raster.gs.geobox, groupby="solar_day", chunks={}
    )

    np.testing.assert_array_equal(
        loaded.start_datetime, [np.datetime64("2025-01-15T10:30:00", "ns")]
    )
    np.testing.assert_array_equal(
        loaded.end_datetime, [np.datetime64("2025-01-15T10:31:40", "ns")]
    )
    assert not loaded.data_vars.keys() & {"start_datetime", "end_datetime"}
    reopened = open_raster(loaded.gs.to_netcdf(tmp_path / "period.nc"))
    xr.testing.assert_equal(reopened.start_datetime, loaded.start_datetime)
    xr.testing.assert_equal(reopened.end_datetime, loaded.end_datetime)


def test_an_item_stating_only_datetime_spans_its_instant(observation):
    item, raster = observation

    loaded = stac.load_dataset([item], geobox=raster.gs.geobox, chunks={})

    assert (loaded.start_datetime == loaded.time).all()
    assert (loaded.end_datetime == loaded.time).all()


def test_bands_group_by_the_grid_they_state_finest_first(observation, tmp_path):
    item, raster = observation
    coarse = _add_coarse_asset(item, raster, tmp_path)

    groups = group_bands_by_grid([item])

    fine = raster.gs.geobox
    assert groups == (
        (fine.crs, fine.resolution, ["red"]),
        (coarse.gs.geobox.crs, coarse.gs.geobox.resolution, ["swir"]),
    )


def test_grouping_skips_the_bands_an_item_lacks(observation, tmp_path):
    item, raster = observation
    _add_coarse_asset(item, raster, tmp_path)
    bare = item.clone()
    del bare.assets["swir"]

    groups = group_bands_by_grid([bare])

    assert [bands for _, _, bands in groups] == [["red"]]


def test_each_native_grid_loads_lazily_without_resampling(observation, tmp_path):
    item, raster = observation
    coarse = _add_coarse_asset(item, raster, tmp_path)

    loaded = stac.load_native_datasets([item], chunks={})

    (_, _, fine_ds), (_, _, coarse_ds) = loaded
    assert fine_ds.gs.geobox == raster.gs.geobox
    assert coarse_ds.gs.geobox == coarse.gs.geobox
    assert list(coarse_ds.data_vars) == ["swir"]
    assert isinstance(coarse_ds.swir.data, da.Array)
    np.testing.assert_array_equal(coarse_ds.swir.isel(time=0), coarse.swir)


def test_a_region_crops_each_native_grid_at_its_own_resolution(observation, tmp_path):
    item, raster = observation
    _add_coarse_asset(item, raster, tmp_path)
    region = raster.gs.geobox[:2, :2].extent.to_crs("EPSG:4326")

    loaded = stac.load_native_datasets([item], chunks={}, geopolygon=region)

    resolutions = [native.gs.geobox.resolution.x for _, _, native in loaded]
    assert resolutions == [10, 20]


def test_property_requests_keep_numbers_exact_and_text_as_text():
    import pystac

    first = pystac.Item(
        "first",
        None,
        None,
        dt(2025, 6, 1, tzinfo=timezone.utc),
        {
            "eo:cloud_cover": 24.001125,
            "platform": "sentinel-2b",
            "start_datetime": "2025-06-01T00:00:00Z",
            "end_datetime": "2025-06-01T00:05:00Z",
            "view:valid": True,
            "instruments": ["msi"],
        },
    )
    second = pystac.Item(
        "second",
        None,
        None,
        dt(2025, 6, 2, tzinfo=timezone.utc),
        {"eo:cloud_cover": 40, "sat:relative_orbit": 8},
    )

    assert stac.list_properties([first, second]) == [
        {"key": "eo:cloud_cover", "dtype": "float64"},
        {"key": "platform", "dtype": "str"},
        {"key": "sat:relative_orbit", "dtype": "float64"},
    ]
