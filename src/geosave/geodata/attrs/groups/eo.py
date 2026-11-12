"""Where in the spectrum one variable was measured, as a STAC eo band says."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import AttrsGroup, Decoded


class EOBand(AttrsGroup):
    """Name the spectral band one variable holds.

    The band name, centre wavelength and width describe its spectral response.

    Attributes:
        common_name: Band name shared across sensors, e.g. `"red"` or `"nir"`.
        center_wavelength: Centre of the band, in micrometres.
        full_width_half_max: Width of the band at half its peak response, in
            micrometres.

    Examples:
        >>> ds["B04"].gs.spectral.common_name
        'red'
    """

    common_name: Annotated[str, Field(min_length=1)] | None = None
    center_wavelength: Decoded[Annotated[float, Field(gt=0)]] | None = None
    full_width_half_max: Decoded[Annotated[float, Field(gt=0)]] | None = None
