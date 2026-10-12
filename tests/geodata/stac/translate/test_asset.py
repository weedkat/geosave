from __future__ import annotations

import pystac

from geosave.geodata.stac.translate import asset


def test_asset_reader_keeps_core_band_precedence() -> None:
    entry = pystac.Asset(
        "https://example.com/a.tif",
        extra_fields={
            "unit": "asset",
            "description": "asset",
            "raster:bands": [{"unit": "raster"}],
            "eo:bands": [{"description": "EO"}],
            "bands": [{"unit": "core", "description": "Core"}],
        },
    )

    assert asset.fields(entry) == {"unit": "core", "description": "Core"}


def test_legacy_listings_pair_by_index() -> None:
    entry = pystac.Asset(
        "a.tif",
        extra_fields={
            "raster:bands": [{"data_type": "uint16", "nodata": 0, "scale": 0.0001}],
            "eo:bands": [{"name": "B04", "common_name": "red"}, {"name": "B08"}],
        },
    )

    (first, red), (second, nir) = asset.bands(entry)

    assert (first.scale, red.common_name) == (0.0001, "red")
    assert (second.to_dict(), nir.name) == ({}, "B08")


def test_core_bands_read_without_parquet_nulls() -> None:
    entry = pystac.Asset(
        "a.tif",
        extra_fields={
            "bands": [{"name": "label", "unit": None, "raster:scale": None}],
            "raster:bands": [{"unit": "ignored"}],
        },
    )

    ((raster, spectral),) = asset.bands(entry)

    assert (raster.to_dict(), spectral.to_dict()) == ({}, {"name": "label"})


def test_an_asset_without_bands_reads_as_none() -> None:
    assert asset.bands(pystac.Asset("thumbnail.png")) == []


def test_a_core_multiband_asset_names_its_variables_by_key() -> None:
    entry = pystac.Asset(
        "visual.tif",
        extra_fields={"bands": [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}]},
    )

    assert asset.names("visual", entry) == ("visual.1", "visual.2", "visual.3")
