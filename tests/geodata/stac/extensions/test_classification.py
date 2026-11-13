"""A variable's legend as the classes its Raster band lists, and back."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import xarray as xr
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.raster import RasterBand

from geosave.geodata import attrs
from geosave.geodata.stac.extensions import classification
from geosave.geodata.stac.stamps import STAMPS, LoadSource
from tests.geodata.conftest import build_raster
from tests.geodata.stac.extensions.conftest import _empty


def _legend(classes, colors=None):
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    return label.gs.write_flags(classes, colors, target=["label"]).label.gs.flags


def test_classes_are_stamped_back_as_the_legend_that_made_them() -> None:
    classes = classification.build_classes(
        _legend({0: "background", 1: "forest"}, {1: "#00ff00"})
    )

    band = RasterBand.create(data_type="uint8")
    ClassificationExtension.ext(band).apply(classes=classes)

    stamped = _empty("label").assign_attrs(classification.read_attrs(band))

    legend = attrs.CFFlags.from_attrs(stamped.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert legend.color_hints[1] == "00FF00"


def _item(id: str = "scene", **asset_fields) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=datetime(2025, 1, 1, tzinfo=UTC),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            media_type=pystac.MediaType.GEOTIFF,
            extra_fields=asset_fields,
        ),
    )
    return item


def _loaded(*names: str) -> xr.Dataset:
    pixels = {}
    for name in names:
        pixels[name] = (("y", "x"), [[1]])
    return xr.Dataset(pixels)


def test_classification_stamp_is_registered() -> None:
    assert STAMPS["classification"] is classification.stamp_classification


def test_classification_stamp_writes_flags_and_colours() -> None:
    item = _item(
        **{
            "raster:bands": [
                {
                    "classification:classes": [
                        {"value": 0, "name": "background"},
                        {"value": 1, "name": "forest", "color_hint": "00FF00"},
                    ]
                }
            ]
        }
    )

    stamped = classification.stamp_classification(
        _loaded("red"), LoadSource([item], None, "id")
    )

    legend = attrs.CFFlags.from_attrs(stamped.red.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert legend.colors[1] == "#00FF00"


def test_classification_stamp_leaves_unclassified_bands_alone() -> None:
    stamped = classification.stamp_classification(
        _loaded("red"), LoadSource([_item()], None, "id")
    )

    assert stamped.red.attrs == {}
