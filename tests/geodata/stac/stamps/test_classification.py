"""Classification flags and colour hints."""

from geosave.geodata import attrs
from geosave.geodata.stac.stamps import stamp_classification
from tests.geodata.stac.stamps.conftest import pixels, scene, with_asset


def test_classification_writes_flags_and_colours() -> None:
    classes = [
        {"value": 0, "name": "background"},
        {"value": 1, "name": "forest", "color_hint": "00FF00"},
    ]
    item = with_asset(
        scene(), "red", **{"raster:bands": [{"classification:classes": classes}]}
    )

    stamped = stamp_classification(pixels("red"), [item])

    legend = attrs.CFFlags.from_attrs(stamped.red.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert legend.colors[1] == "#00FF00"


def test_classification_leaves_unclassified_bands_alone() -> None:
    item = with_asset(scene(), "red")

    stamped = stamp_classification(pixels("red"), [item])

    assert stamped.red.attrs == {}
