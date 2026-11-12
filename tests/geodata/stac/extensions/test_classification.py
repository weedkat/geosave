"""A variable's legend as the classes its Raster band lists, and back."""

from __future__ import annotations

import pytest
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.raster import RasterBand

from geosave.geodata import attrs
from geosave.geodata.stac.extensions import classification
from tests.geodata.conftest import build_raster
from tests.geodata.stac.extensions.conftest import _empty


def _legend(classes, colors=None):
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    return label.gs.write_flags(classes, colors, target=["label"]).label.gs.flags


def test_a_class_name_stac_refuses_raises() -> None:
    with pytest.raises(ValueError, match="tree/shrub"):
        classification.build_classes(_legend({1: "tree/shrub"}))


def test_bit_masks_have_no_classes() -> None:
    qa = build_raster().nir.astype("uint8")
    qa.attrs.update(flag_masks=[1, 2], flag_meanings="cloud shadow")

    assert classification.build_classes(qa.gs.flags) is None


def test_classes_are_stamped_back_as_the_legend_that_made_them() -> None:
    classes = classification.build_classes(
        _legend({0: "background", 1: "forest"}), attrs.Display(color_map={1: "#00ff00"})
    )

    band = RasterBand.create(data_type="uint8")
    ClassificationExtension.ext(band).apply(classes=classes)

    stamped = _empty("label").assign_attrs(classification.read_attrs(band))

    legend = attrs.CFFlags.from_attrs(stamped.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert attrs.Display.from_attrs(stamped.attrs).color_map[1] == "#00FF00"
