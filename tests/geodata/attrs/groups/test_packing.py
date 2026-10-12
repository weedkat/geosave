from __future__ import annotations

import pytest

from geosave_engine.geodata import attrs


def test_packing_spells_cf_scale_and_offset() -> None:
    packing = attrs.Packing(scale_factor=1e-4, add_offset=-0.1)

    assert packing.model_dump(by_alias=True, exclude_none=True) == {
        "scale_factor": 1e-4,
        "add_offset": -0.1,
    }


def test_an_offset_alone_scales_by_one() -> None:
    assert attrs.Packing.model_validate({"add_offset": 1}).scale_factor == 1.0


def test_text_spelled_packing_reads_as_numbers() -> None:
    assert attrs.Packing.model_validate({"scale_factor": "0.0001"}).scale_factor == 1e-4


def test_a_zero_scale_is_refused() -> None:
    with pytest.raises(ValueError):
        attrs.Packing(scale_factor=0)


def test_the_keys_cover_what_it_writes() -> None:
    written = attrs.Packing(scale_factor=2, add_offset=1).model_dump(by_alias=True)
    assert set(written) <= set(attrs.Packing.KEYS)
