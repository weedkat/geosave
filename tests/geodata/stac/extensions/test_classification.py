"""A variable's legend as the classes its Raster band lists, and back."""

from __future__ import annotations

import pytest

from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.raster import RasterBand

from geosave.geodata import attrs
from geosave.geodata.stac.extensions import classification
from tests.geodata.conftest import build_raster
from tests.geodata.stac.extensions.conftest import _empty


def _label(classes, colors=None):
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    return label.gs.write_legend(classes, colors, target="label").label


def test_a_legend_becomes_classes() -> None:
    classes = classification.from_dataarray(
        _label({0: "background", 1: "forest"}, {1: "#00ff00"})
    )

    assert [(entry.value, entry.name) for entry in classes] == [
        (0, "background"),
        (1, "forest"),
    ]
    assert classes[1].color_hint == "00FF00"


def test_a_variable_without_a_legend_has_no_classes(scene) -> None:
    assert classification.from_dataarray(scene.red) is None


def test_a_class_name_stac_refuses_raises() -> None:
    with pytest.raises(ValueError, match="tree/shrub"):
        classification.from_dataarray(_label({1: "tree/shrub"}))


def test_bit_masks_have_no_classes() -> None:
    qa = build_raster().nir.astype("uint8")
    qa.attrs.update(flag_masks=[1, 2], flag_meanings="cloud shadow")

    assert classification.from_dataarray(qa) is None


def test_classes_are_stamped_back_as_the_legend_that_made_them() -> None:
    classes = classification.from_dataarray(
        _label({0: "background", 1: "forest"}, {1: "#00ff00"})
    )

    band = RasterBand.create(data_type="uint8")
    ClassificationExtension.ext(band).apply(classes=classes)

    stamped = classification.stamp(_empty("label"), band)

    legend = attrs.Legend.model_validate(stamped.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert legend.color_map[1] == "#00FF00"


def test_a_band_without_classes_stamps_nothing() -> None:
    variable = _empty("label")

    assert (
        classification.stamp(variable, RasterBand.create(data_type="uint8")) is variable
    )
