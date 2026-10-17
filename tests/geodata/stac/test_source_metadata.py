from __future__ import annotations

import subprocess
import sys
from datetime import datetime as dt
from datetime import timezone

import pystac
import pytest
import xarray as xr

import geosave.geodata  # noqa: F401  — registers the .gs accessor
from geosave.geodata.attrs import Packing, Spectral
from unittest.mock import Mock

from geosave.geodata.stac.source import StacSource
from geosave.geodata.stac.search.api import StacClient
from geosave.geodata.warnings import DroppedAttrsWarning


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


def test_importing_attrs_does_not_import_stac() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geosave.geodata.attrs; "
            "assert 'geosave.geodata.stac' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
