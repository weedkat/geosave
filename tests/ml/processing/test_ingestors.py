"""Sentinel-2 ingestion keeps one processing per acquisition and decodes by baseline."""

from datetime import datetime

import pystac
import pytest

from geosave.ml.processing.ingestors import drop_superseded, read_esa_packing


def _item(item_id: str, baseline: str = "05.00") -> pystac.Item:
    return pystac.Item(
        item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 7, 1),
        properties={"s2:processing_baseline": baseline},
    )


def test_only_the_newest_processing_of_an_acquisition_is_kept():
    original = _item("S2A_MSIL2A_20200101T000000_R001_T48MXU_20200101T120000")
    reprocessed = _item("S2A_MSIL2A_20200101T000000_R001_T48MXU_20230101T120000")
    other = _item("S2A_MSIL2A_20200101T000000_R001_T48MXV_20200101T120000")

    kept = drop_superseded([reprocessed, original, other])

    assert kept == [reprocessed, other]


@pytest.mark.parametrize(("baseline", "offset"), [("05.00", -0.1), ("02.12", 0.0)])
def test_the_processing_baseline_sets_the_reflectance_offset(baseline, offset):
    packing = read_esa_packing([_item("a", baseline)])

    assert (packing.scale_factor, packing.add_offset) == (1e-4, offset)


def test_tiles_of_mixed_baselines_are_refused():
    with pytest.raises(ValueError, match="mix processing baselines"):
        read_esa_packing([_item("a", "05.00"), _item("b", "02.12")])
