from __future__ import annotations

from datetime import datetime as dt
from datetime import timezone

import dask.array as da
import numpy as np
import pystac
import pytest
import xarray as xr

import geosave.geodata  # noqa: F401  — registers the .gs accessor
from geosave.geodata import open_raster, open_vector
from geosave.geodata.attrs import Packing, Spectral
from geosave.geodata.features import ndvi
from geosave.geodata.stac.load import load_dataset
from geosave.geodata.stac import StacQuery
from tests.geodata.conftest import create_file_items, create_test_collection


def test_effective_nodata_follows_loaded_pixels(local_catalog):
    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    loaded = load_dataset(
        matched,
        geobox=grid,
        collection=collection,
        **{**options, "dtype": "float32", "nodata": -9999},
    )

    assert isinstance(loaded.red.data, da.Array)
    assert loaded.red.values[0, 0, 0] == -9999
    assert loaded.red.attrs["_FillValue"] == -9999
    prepared = loaded.gs.mask_and_scale()
    assert np.isnan(prepared.red.values[0, 0, 0])
    assert prepared.red.values[0, 1, 1] == pytest.approx(0.2)


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
    assert loaded["red.1"].gs.packing == Packing(scale_factor=0.1)
    assert loaded["red.2"].attrs == {"units": "K", "long_name": "Temperature"}
    assert item.to_dict() == before


def test_loader_packing_overrides_the_agreed_source_packing() -> None:
    item = _item("scene", dt(2025, 1, 1), {"scale": 0.1, "offset": 2})
    loaded = xr.Dataset(
        {"red": xr.DataArray([[1]], dims=("y", "x"), attrs={"scale_factor": 1.0})}
    )

    loaded = _apply_metadata([item], _collection(), loaded)

    assert loaded.red.gs.packing == Packing(scale_factor=1, add_offset=2)


def test_spectral_facts_reach_the_variable() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]
    }

    loaded = _apply_metadata(
        [item], _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.gs.spectral == Spectral(
        common_name="red", center_wavelength=0.665
    )


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


def test_loaded_properties_fuse_and_publish_through_geoparquet(local_catalog, tmp_path):

    import xarray as xr

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    first = matched[0]
    first.properties["eo:cloud_cover"] = 20.0
    second = first.clone()
    second.id = "second"
    second.properties["eo:cloud_cover"] = 40.0
    matched = [first, second]
    loaded = load_dataset(
        matched,
        geobox=grid,
        collection=collection,
        **{**options, "with_properties": ["eo:cloud_cover"]},
    )
    # Two scenes of one date: odc-stac fuses their values into that date's one.
    assert loaded.gs.stac_properties["eo:cloud_cover"].tolist() == [30.0]
    assert loaded.red.chunks is not None
    file = loaded.gs.to_netcdf(tmp_path / "property.nc")
    with xr.open_dataset(file) as stored:
        assert stored.gs.stac_properties["eo:cloud_cover"].tolist() == [30.0]
    output = create_test_collection("processed", description="Processed observations")
    rows = loaded.gs.to_catalog(
        tmp_path, split_asset=(), format_paths="processed.tif", collection=output
    )
    saved = rows.gs.to_geoparquet(tmp_path / "processed.parquet")
    assert open_vector(saved).gs.to_items()[0].properties["eo:cloud_cover"] == 30.0


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
        create_file_items(
            loaded.gs.to_cog(tmp_path, split_asset=(), format_paths="composite.tif")
        )
    )
    assert str(loaded.start_datetime.values[0])[:10] == "2025-01-01"
    assert str(loaded.end_datetime.values[0])[:19] == "2025-01-31T23:59:59"
    assert rebuilt.properties["start_datetime"].startswith("2025-01-01")
    assert rebuilt.properties["end_datetime"].startswith("2025-01-31T23:59:59")


def test_loaded_stac_metadata_is_interpretable_by_cf_without_pixel_reads(local_catalog):
    from dask.callbacks import Callback

    client, region, grid, options = local_catalog
    matched = client.search_items(StacQuery(collections=["local"]).set_region(region))
    collection = client.collection("local")
    with Callback(
        pretask=lambda *_: pytest.fail("metadata inspection computed pixels")
    ):
        loaded = load_dataset(matched, geobox=grid, collection=collection, **options)
        assert loaded.cf.axes["X"] == ["x"]
        assert loaded.cf.axes["Y"] == ["y"]
        assert loaded.cf.coordinates["time"] == ["time"]
        mappings = loaded.cf.grid_mappings
        assert len(mappings) == 1
        assert mappings[0].crs == grid.crs.proj
        assert loaded.red.gs.packing.scale_factor == 0.0001
        assert loaded.nir.gs.packing.scale_factor == 0.0002
        assert "nodata" not in loaded.red.attrs
        assert loaded.gs.stac_collection.id == collection.id
