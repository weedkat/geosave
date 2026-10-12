from __future__ import annotations

import pytest

from geosave_engine.geodata import attrs


def test_spectral_spells_its_three_keys() -> None:
    spectral = attrs.Spectral(common_name="red", center_wavelength=0.665)

    assert spectral.model_dump(by_alias=True, exclude_none=True) == {
        "common_name": "red",
        "center_wavelength": 0.665,
    }


def test_text_wavelengths_read_as_numbers() -> None:
    assert (
        attrs.Spectral.model_validate({"center_wavelength": "0.665"}).center_wavelength
        == 0.665
    )


def test_a_non_positive_wavelength_is_refused() -> None:
    with pytest.raises(ValueError):
        attrs.Spectral(center_wavelength=0)


def test_the_keys_cover_what_it_writes() -> None:
    written = attrs.Spectral(
        common_name="red", center_wavelength=0.665, full_width_half_max=0.03
    ).model_dump(by_alias=True)
    assert set(written) <= set(attrs.Spectral.KEYS)
