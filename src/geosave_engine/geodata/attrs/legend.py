"""What a label variable's pixel values mean, and how they colour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from pydantic import model_validator

from geosave_engine.geodata.utils.color import Palette

from .base import AttrsGroup, Decoded, Join

if TYPE_CHECKING:
    import xarray as xr


class Legend(AttrsGroup):
    """Say what a label variable's pixel values mean, and how they colour.

    CF lists classes two ways: `flag_values` enumerates the values a pixel
    takes and `flag_masks` the bits a value packs flags into; `flag_meanings`
    names them in order. `classes` reads an enumeration back keyed by value.

    Args:
        flag_values: Pixel values, ascending and unique, one per class.
        flag_masks: Bit masks, for values packing several flags at once.
        flag_meanings: Space-separated class names, in listing order.
        color_map: `{pixel value: hex or RGB}`, keyed to listed values.

    Raises:
        ValueError: Names and codes differ in number or only one is set,
            `flag_values` is not ascending and unique, a class name carries
            whitespace, a colour keys to an unlisted value, or a code does
            not fit the band's dtype.

    Examples:
        >>> ds["label"].gs.legend.classes
        {0: 'bg', 1: 'palm'}
    """

    JOIN: ClassVar[Join] = "agree"

    flag_values: Decoded[list[int]] | None = None
    flag_masks: Decoded[list[int]] | None = None
    flag_meanings: str | None = None
    color_map: Decoded[Palette] | None = None

    @property
    def classes(self) -> dict[int, str] | None:
        """Return each pixel value mapped to its class, None for a bitfield or colours only."""
        if self.flag_values is None or self.flag_meanings is None:
            return None
        return dict(zip(self.flag_values, self.flag_meanings.split(), strict=True))

    @model_validator(mode="before")
    @classmethod
    def _spell_classes(cls, data: Any) -> Any:
        """Accept `classes` in place of the codes and names CF stores."""
        if not isinstance(data, dict) or data.get("classes") is None:
            return data
        given = dict(data)
        classes = given.pop("classes")
        split = sorted(name for name in classes.values() if name.split() != [name])
        if split:
            raise ValueError(
                f"class names {split} are empty or carry whitespace; "
                f"flag_meanings tokens are single words, so use '_'"
            )
        values = sorted(classes)
        given["flag_values"] = values
        given["flag_meanings"] = " ".join(classes[value] for value in values)
        return given

    @model_validator(mode="after")
    def _line_up(self) -> Self:
        """Refuse a listing whose names and codes do not describe one another."""
        listed = {
            name: codes
            for name, codes in (
                ("flag_values", self.flag_values),
                ("flag_masks", self.flag_masks),
            )
            if codes is not None
        }
        if self.flag_meanings is None:
            if listed:
                raise ValueError(
                    f"{sorted(listed)} list codes but flag_meanings names no classes for them"
                )
        elif not listed:
            raise ValueError(
                "flag_meanings names classes that no flag_values or flag_masks enumerate"
            )
        else:
            names = self.flag_meanings.split()
            for name, codes in listed.items():
                if len(codes) != len(names):
                    raise ValueError(
                        f"{name} lists {len(codes)} codes but flag_meanings names "
                        f"{len(names)} classes"
                    )
        if (
            self.flag_values is not None
            and sorted(set(self.flag_values)) != self.flag_values
        ):
            raise ValueError(
                f"flag_values {self.flag_values} are not ascending and unique"
            )
        # A colour for a value the listing does not name paints nothing.
        if self.color_map is not None and self.flag_values is not None:
            unlisted = sorted(set(self.color_map) - set(self.flag_values))
            if unlisted:
                raise ValueError(
                    f"colours {unlisted} key to values this listing does not name; "
                    f"it lists {self.flag_values}"
                )
        return self

    def held_by(self, band: xr.DataArray) -> Legend:
        """Return the legend, refusing a code `band` cannot hold exactly.

        A code the band stores as another number would name another class.

        Raises:
            ValueError: A listed code is no value of the band's dtype.
        """
        codes = [*(self.flag_values or ()), *(self.flag_masks or ())]
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(codes).astype(band.dtype)
        outside = [
            code for code, kept in zip(codes, stored, strict=True) if kept != code
        ]
        if outside:
            raise ValueError(f"codes {outside} are no {band.dtype} values")
        return self


def is_flag(band: xr.DataArray) -> bool:
    """Say whether a variable's values are class codes, not measurements.

    Dtype does not say: Sentinel-2 reflectance and a land cover map are both
    integers. A legend key or a GDAL palette marks the values as codes, which
    blending or ranking would destroy. Only the keys are read, so attrs that
    fail `Legend` validation still answer.

    Examples:
        >>> is_flag(scene.landcover), is_flag(scene.red)
        (True, False)
    """
    return (
        any(key in band.attrs for key in Legend.KEYS)
        or band.attrs.get("colorinterp") == "palette"
    )
