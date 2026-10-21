from __future__ import annotations

from datetime import datetime as dt
from datetime import timezone
from typing import Any
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from pydantic import ValidationError

import geosave.geodata  # noqa: F401  — registers the .gs accessor
import geosave.geodata.stac.source as source_module
from geosave.geodata import GeoAnchor, open_raster, open_vector, raster, stac
from geosave.geodata.attrs import Packing, Spectral
from geosave.geodata.features import ndvi
from geosave.geodata.stac import StacQuery, StacSource, StacSourceConfig
from geosave.geodata.stac.search.api import StacClient
from geosave.geodata.warnings import DroppedAttrsWarning


class FakeClient:
    def search_items(self, query: object) -> list[pystac.Item]:
        return [
            pystac.Item(
                "scene", None, None, dt.fromisoformat("2025-06-01T00:00:00+00:00"), {}
            )
        ]

    def collection(self, collection: str) -> object:
        return object()


def _target() -> GeoAnchor:
    return GeoAnchor.from_coordinates(
        -6.5914,
        107.8416,
        shape=2,
        resolution=10,
        timespan="2025-06",
    )


def test_search_query_preserves_explicit_recipe_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    query = StacQuery(
        collections=["example"],
        intersects={"type": "Point", "coordinates": [107.8, -6.5]},
        datetime="2024-01",
    )
    source.query = query

    assert source._search_query(_target()) == query


def test_search_query_preserves_explicit_bbox() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    source.query = StacQuery(collections=["example"], bbox=(1, 2, 3, 4))

    query = source._search_query(_target())

    assert query.bbox == (1, 2, 3, 4)
    assert query.datetime == _target().timespan


def test_search_query_fills_omitted_target_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]

    query = source._search_query(_target())

    assert query.bbox is not None
    assert query.datetime == _target().timespan


def test_item_ids_do_not_gain_target_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    query = StacQuery(collections=["example"], ids=["scene-1"])
    source.query = query

    assert source._search_query(_target()) == query


def test_source_config_defaults_to_spatial_dask_chunks() -> None:
    first = StacSourceConfig()
    second = StacSourceConfig()

    assert first.chunks == {"x": 1024, "y": 1024}
    assert first.chunks is not second.chunks


def test_source_config_accepts_mixed_chunk_sizes() -> None:
    configured = StacSourceConfig(chunks={"x": 1024, "y": "auto"})

    assert configured.chunks == {"x": 1024, "y": "auto"}


def test_source_config_keeps_one_kernel_per_band() -> None:
    configured = StacSourceConfig(resampling={"B04": "bilinear", "SCL": "nearest"})

    assert configured.resampling == {"B04": "bilinear", "SCL": "nearest"}


def test_source_config_refuses_a_kernel_nested_under_a_band() -> None:
    with pytest.raises(ValidationError, match="resampling"):
        StacSourceConfig(resampling={"B04": {"x": "nearest"}})


def test_source_config_refuses_an_unknown_kernel() -> None:
    with pytest.raises(ValidationError, match="resampling"):
        StacSourceConfig(resampling="not-a-kernel")


def test_load_returns_dask_backed_arrays_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(items: list[object], *, geobox: GeoBox, **options: Any) -> xr.Dataset:
        captured.update(options)
        values = np.ones((1, *geobox.shape), dtype="uint16")
        loaded = raster(
            {
                "red": (
                    ("time", *("y", "x")),
                    values,
                )
            },
            geobox,
            time=[dt(2025, 6, 1)],
        )
        return loaded.chunk(options["chunks"])

    monkeypatch.setattr(source_module.odc.stac, "load", fake_load)
    monkeypatch.setattr(
        StacSource, "_apply_metadata", lambda self, items, loaded: loaded
    )
    anchor = GeoAnchor.from_coordinates(
        -6.5914,
        107.8416,
        shape=2,
        resolution=10,
        timespan="2025-06",
    )

    loaded = StacSource(FakeClient(), collection="example").load(anchor)  # type: ignore[arg-type]

    assert captured["chunks"] == {"x": 1024, "y": 1024}
    assert isinstance(loaded["red"].data, da.Array)


def test_load_names_a_geographic_grid_y_x(monkeypatch: pytest.MonkeyPatch) -> None:
    from odc.geo.xr import xr_coords

    def fake_load(items: list[object], *, geobox: GeoBox, **options: Any) -> xr.Dataset:
        # odc-stac names a geographic grid latitude/longitude.
        coords = dict(xr_coords(geobox))
        coords["time"] = [dt(2025, 6, 1)]
        loaded = xr.Dataset(
            {
                "red": (
                    ("time", *geobox.dimensions),
                    da.ones((1, *geobox.shape), chunks=-1),
                )
            },
            coords=coords,
        )
        loaded.red.encoding["grid_mapping"] = "spatial_ref"
        return loaded

    monkeypatch.setattr(source_module.odc.stac, "load", fake_load)
    monkeypatch.setattr(
        StacSource, "_apply_metadata", lambda self, items, loaded: loaded
    )
    anchor = GeoAnchor.from_coordinates(
        -6.5914,
        107.8416,
        shape=2,
        resolution=0.0001,
        crs="EPSG:4326",
        timespan="2025-06",
    )
    assert anchor.geobox.dimensions == ("latitude", "longitude")

    loaded = StacSource(FakeClient(), collection="example").load(anchor)  # type: ignore[arg-type]

    assert loaded.red.dims == ("time", "y", "x")
    assert loaded.gs.geobox == anchor.geobox
    assert isinstance(loaded.red.data, da.Array)
    assert loaded.y.attrs["standard_name"] == "latitude"


def test_chunks_none_remains_an_explicit_eager_mode() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]

    source = source.set_config(chunks=None)

    assert source.config.chunks is None
    assert source.config.to_load_kwargs()["chunks"] is None


def test_with_properties_forwards_odc_native_requests() -> None:
    source = StacSource(FakeClient(), collection="example").set_config(  # type: ignore[arg-type]
        with_properties=(
            "platform",
            {
                "key": "view:sun_azimuth",
                "name": "view:sun_azimuth",
                "dtype": "float32",
                "units": "degree",
            },
        )
    )

    assert source.config.to_load_kwargs()["with_properties"] == [
        {"key": "platform", "name": "platform"},
        {
            "key": "view:sun_azimuth",
            "name": "view:sun_azimuth",
            "dtype": "float32",
            "units": "degree",
        },
    ]


def test_effective_nodata_follows_loaded_pixels(local_source):
    source, anchor = local_source
    loaded = source.set_config(dtype="float32", nodata=-9999).load(anchor)

    assert isinstance(loaded.red.data, da.Array)
    assert loaded.red.values[0, 0, 0] == -9999
    assert loaded.red.attrs["_FillValue"] == -9999
    prepared = loaded.gs.to_nan().gs.unpack()
    assert np.isnan(prepared.red.values[0, 0, 0])
    assert prepared.red.values[0, 1, 1] == pytest.approx(0.2)


def test_aliases_resolve_each_multiband_asset_index(local_source):
    source, anchor = local_source
    loaded = source.set_config(chunks=None).load(anchor)

    assert isinstance(loaded.red.data, np.ndarray)
    assert loaded.red.attrs["scale_factor"] == 0.0001
    assert loaded.nir.attrs["scale_factor"] == 0.0002
    assert loaded.nir.attrs["add_offset"] == -0.1
    assert loaded.gs.geobox == anchor.geobox
    assert loaded.gs.to_nan().gs.unpack().nir.values[0, 1, 1] == pytest.approx(1.1)


def test_a_source_searches_the_items_it_loads_and_keeps_no_copy(local_source) -> None:
    source, anchor = local_source

    rows = source.search(anchor)
    loaded = source.load(anchor)

    assert type(rows.gs).__name__ == "GeoCatalog"
    assert source.collection.id == rows["collection"].iloc[0]
    assert (
        rows.gs.collections[source.collection.id].description
        == source.collection.description
    )
    assert "stac_items" not in loaded.attrs
    assert "stac_groupby" not in loaded.attrs


def test_a_load_states_its_collection_as_acdd(local_source) -> None:
    source, anchor = local_source
    loaded = source.load(anchor)

    assert loaded.attrs["id"] == source.collection.id
    assert loaded.attrs["license"] == source.collection.license
    assert loaded.attrs["summary"] == "Local reflectance"


def test_a_one_item_query_keeps_odcs_time_axis(local_source) -> None:
    source, anchor = local_source

    many = source.load(anchor)
    one = source.set_query(max_items=1).load(anchor)

    assert "time" in many.red.dims
    assert one.sizes["time"] == 1
    assert one.red.dims == many.red.dims
    assert one.squeeze("time").time.ndim == 0


def test_narrowing_returns_a_copy_and_leaves_the_source_unchanged() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]

    red = source.set_config(bands=["red"])
    one = red.set_query(max_items=1)

    assert source.config.bands is None and source.query.max_items is None
    assert red.config.bands == ["red"] and red.query.max_items is None
    assert one.config.bands == ["red"] and one.query.max_items == 1


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
def test_source_keeps_odc_time_and_property_representation(
    local_source, groupby, properties
):
    from copy import deepcopy
    from unittest.mock import Mock

    import odc.stac

    source, anchor = local_source
    composite = source.search(anchor).gs.to_items()[0]
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
    collection = source.collection
    source.client = Mock()
    source.client.search_items.return_value = [composite, instant]
    source.client.collection.return_value = collection
    original = deepcopy(instant.properties)

    configured = source.set_config(
        groupby=groupby,
        with_properties=properties,
    )
    expected = odc.stac.load(
        [composite, instant], geobox=anchor.geobox, **configured.config.to_load_kwargs()
    )
    loaded = configured.load(anchor)

    assert isinstance(loaded.red.data, da.Array)
    assert dict(loaded.sizes) == dict(expected.sizes)
    assert set(loaded.coords) == set(expected.coords)
    assert set(loaded.data_vars) == set(expected.data_vars)
    xr.testing.assert_equal(loaded.time, expected.time)
    np.testing.assert_array_equal(loaded.red.values, expected.red.values)
    for request in properties or []:
        key = request if isinstance(request, str) else request["key"]
        xr.testing.assert_equal(loaded[key], expected[key])
    assert instant.properties == original
    if not properties:
        assert "start_datetime" not in loaded
        assert "end_datetime" not in loaded


def _apply_metadata(items, collection, loaded):
    client = Mock(spec=StacClient)
    client.collection.return_value = collection
    source = StacSource(client, collection=collection.id)
    return source._apply_metadata(items, loaded)


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


def test_reading_uses_band_description_not_common_name() -> None:
    timestamp = dt(2025, 1, 1)
    item = _item("scene-1", timestamp, {"unit": "1"})
    item.assets["red"].extra_fields["eo:bands"] = [
        {"name": "B04", "common_name": "red", "description": "Surface reflectance"}
    ]

    loaded = _apply_metadata(
        [item], _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.attrs["long_name"] == "Surface reflectance"
    assert loaded.red.attrs["units"] == "1"


@pytest.mark.parametrize("second", [{}, {"scale": 0.0002}])
def test_disagreeing_packing_drops_with_a_warning(second) -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"scale": 0.0001}),
        _item("scene-2", dt(2025, 1, 2), second),
    ]

    with pytest.warns(DroppedAttrsWarning, match="scale_factor") as warned:
        loaded = _apply_metadata(
            items, _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
        )

    assert loaded.red.gs.packing is None
    assert "'scene-2'" in str(warned[0].message)


def test_an_item_without_the_asset_drops_its_fields_with_a_warning() -> None:
    first = _item("scene-1", dt(2025, 1, 1), {"unit": "1"})
    second = _item("scene-2", dt(2025, 1, 2), {"unit": "1"})
    del second.assets["red"]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        loaded = _apply_metadata(
            [first, second],
            _collection(),
            xr.Dataset({"red": (("y", "x"), [[1]])}),
        )

    assert "units" not in loaded.red.attrs


def test_reading_writes_no_packing_when_no_item_publishes_it() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {"unit": "1"}),
    ]

    loaded = _apply_metadata(
        items, _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
    )

    assert loaded.red.gs.packing is None


def test_reading_drops_a_label_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        loaded = _apply_metadata(
            items, _collection(), xr.Dataset({"red": (("y", "x"), [[1]])})
        )

    assert "units" not in loaded.red.attrs


def test_reading_rejects_an_empty_item_sequence() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        _apply_metadata([], _collection(), xr.Dataset({"red": (("y", "x"), [[1]])}))


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


def test_observations_to_prepared_index_to_reusable_dataset(local_source, tmp_path):
    source, anchor = local_source
    raw = source.load(anchor)
    reflectance = raw.gs.mask_and_scale()
    derived = reflectance.assign(ndvi=ndvi(reflectance, nir="nir", red="red"))[["ndvi"]]
    composite = derived.median("time", keep_attrs=True)
    assert isinstance(composite.ndvi.data, da.Array)
    assert composite.gs.geobox == anchor.geobox
    written = composite.gs.to_zarr(tmp_path / "ndvi.zarr")

    with open_raster(written, chunks={}) as reopened:
        assert isinstance(reopened.ndvi.data, da.Array)
        assert reopened.gs.geobox == anchor.geobox
        assert np.isnan(reopened.ndvi.values[0, 0])
        assert reopened.ndvi.values[1, 1] == pytest.approx((1.1 - 0.2) / (1.1 + 0.2))
        assert "scale_factor" not in reopened.ndvi.attrs
    # Preparation leaves the reusable source configuration and packed input intact.
    assert raw.red.attrs["scale_factor"] == 0.0001
    assert raw.red.values[0, 1, 1] == 2000


def test_property_rename_is_rejected():
    from geosave.geodata.stac.source import StacSourceConfig

    with pytest.raises(ValueError, match="exact STAC key"):
        StacSourceConfig(with_properties=[{"key": "eo:cloud_cover", "name": "cloud"}])


def test_native_collection_property_does_not_search():
    from unittest.mock import Mock

    from geosave.geodata.stac.source import StacSource

    collection = stac.create_collection("optical", description="Reflectance")
    client = Mock()
    client.collection.return_value = collection
    source = StacSource(client, collection="optical")
    assert source.collection is collection
    client.search_items.assert_not_called()


def test_source_properties_fuse_and_publish_through_geoparquet(local_source, tmp_path):
    from unittest.mock import Mock

    import numpy as np
    import xarray as xr

    source, anchor = local_source
    collection = source.collection
    first = source.search(anchor).gs.to_items()[0]
    first.properties["eo:cloud_cover"] = 20.0
    second = first.clone()
    second.id = "second"
    second.properties["eo:cloud_cover"] = 40.0
    client = Mock()
    client.collection.return_value = collection
    client.search_items.return_value = [first, second]
    source.client = client
    loaded = source.set_config(with_properties=["eo:cloud_cover"]).load(anchor)
    loaded = loaded.set_coords("eo:cloud_cover")
    assert loaded["eo:cloud_cover"].dims == ("time",)
    assert loaded["eo:cloud_cover"].item() == 30.0
    assert loaded.red.chunks is not None
    file = loaded.gs.to_netcdf(tmp_path / "property.nc")
    with xr.open_dataset(file.written["image"].path) as stored:
        np.testing.assert_equal(stored["eo:cloud_cover"].values, [30.0])
    scenes = loaded.gs.to_cog(tmp_path / "processed.tif")
    output = stac.create_collection("processed", description="Processed observations")
    rows = stac.create_catalog(
        stac.create_items(scenes, collection=output), collections=[output]
    )
    saved = rows.gs.to_geoparquet(tmp_path / "processed.parquet")
    assert open_vector(saved).gs.to_items()[0].properties["eo:cloud_cover"] == 30.0


def test_stac_source_does_not_invent_composite_bounds(local_source, tmp_path):
    from unittest.mock import Mock

    source, anchor = local_source
    item = source.search(anchor).gs.to_items()[0]
    item.properties.update(
        start_datetime="2025-01-01T00:00:00Z", end_datetime="2025-01-31T23:59:59Z"
    )
    collection = source.collection
    source.client = Mock()
    source.client.search_items.return_value = [item]
    source.client.collection.return_value = collection
    loaded = source.load(anchor)
    rebuilt = next(stac.create_items(loaded.gs.to_cog(tmp_path / "composite.tif")))
    assert "start_datetime" not in loaded
    assert "end_datetime" not in loaded
    assert "start_datetime" not in rebuilt.properties
    assert "end_datetime" not in rebuilt.properties
