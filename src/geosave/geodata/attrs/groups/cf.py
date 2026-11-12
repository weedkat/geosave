"""CF variable attributes: description (ch. 3), flags (§3.5), packing (§8.1, §2.5.1)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import AliasChoices, Field, field_validator, model_validator
from xarray.core.utils import equivalent

from .base import AttrsGroup, Decoded, decode


class CFDescription(AttrsGroup):
    """Say what one variable's values are, as CF chapter 3 describes data.

    Find variables by these names with `.cf.standard_names` or
    `ds.cf["<standard name>"]`; this group reads them for one variable.

    Attributes:
        long_name: Human-readable name, as `"Red reflectance"`.
        standard_name: CF standard name, as `"surface_bidirectional_reflectance"`.
        units: UDUNITS unit, as `"1"` or `"m"`.

    Examples:
        >>> ds["B04"].gs.description.units
        '1'
    """

    long_name: str | None = None
    standard_name: str | None = None
    units: str | None = None


class CFPacking(AttrsGroup):
    """State how one variable's stored values decode to physical values.

    CF packed data (§8.1) with its missing data (§2.5.1): a packed fill is
    given in the stored type, so the three decode together, as
    `xr.decode_cf(mask_and_scale=True)` reads them.

    These fields live in attrs while pixels hold stored values. Native CF
    decoding moves consumed fields into encoding for the reverse conversion.

    Attributes:
        scale_factor: Multiplier applied to each stored value, never 0.
        add_offset: Value added after scaling.
        fill_value: Stored missing-value sentinel, written as `_FillValue`.
            ODC's `nodata` spelling is also accepted when reading.
        missing_value: Additional CF missing-value code or codes.
        valid_min: Smallest valid stored value.
        valid_max: Largest valid stored value.
        valid_range: Smallest and largest valid stored values.

    Examples:
        >>> ds["B04"].gs.packing.scale_factor
        0.0001
    """

    scale_factor: Decoded[float] = 1.0
    add_offset: Decoded[float] = 0.0
    fill_value: Decoded[int | float] | None = Field(
        default=None,
        validation_alias=AliasChoices("_FillValue", "nodata"),
        serialization_alias="_FillValue",
    )
    missing_value: Decoded[int | float | list[int | float]] | None = None
    valid_min: Decoded[int | float] | None = None
    valid_max: Decoded[int | float] | None = None
    valid_range: Decoded[list[int | float]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _check_fill_aliases(cls, data: Any) -> Any:
        """Reject disagreeing CF and ODC spellings of the same sentinel."""
        if isinstance(data, dict) and {"_FillValue", "nodata"} <= data.keys():
            cf, odc = decode(data["_FillValue"]), decode(data["nodata"])
            if not equivalent(cf, odc):
                raise ValueError(
                    f"The attrs give _FillValue as {cf!r} but nodata as {odc!r}; "
                    "both must describe the same missing-value code."
                )
        return data

    @field_validator("scale_factor")
    @classmethod
    def _check_scale_factor_not_zero(cls, scale_factor: float) -> float:
        """Refuse a zero scale factor, which decodes every stored value to the offset."""
        if scale_factor == 0:
            raise ValueError(
                "scale_factor cannot be 0, because every stored value would decode "
                "to the same number. Use the multiplier that converts stored values "
                "to physical ones, such as 0.0001 for Sentinel-2 reflectance."
            )
        return scale_factor


class CFFlags(AttrsGroup):
    """Say what a label variable's pixel values mean, as CF flags (§3.5).

    CF lists classes two ways: `flag_values` enumerates the values a pixel
    takes and `flag_masks` the bits a value packs flags into; `flag_meanings`
    names them in order. `classes` reads an enumeration back keyed by value.

    Attributes:
        flag_values: Enumeration codes, ascending and unique without masks;
            masked codes may repeat when their masks differ.
        flag_masks: Bit masks, for values packing several flags at once.
        flag_meanings: Space-separated class names, in listing order.

    Raises:
        ValueError: Names and codes differ in number or only one is set,
            `flag_values` is not ascending and unique, a class name carries
            whitespace, or a code does not fit the band's dtype.

    Examples:
        >>> ds["label"].gs.flags.classes
        {0: 'bg', 1: 'palm'}
    """

    flag_values: Decoded[list[int]] | None = None
    flag_masks: Decoded[list[int]] | None = None
    flag_meanings: str | None = None

    @property
    def classes(self) -> dict[int, str] | None:
        """Return each pixel value mapped to its class, None for a bitfield or colours only."""
        if (
            self.flag_masks is not None
            or self.flag_values is None
            or self.flag_meanings is None
        ):
            return None
        return dict(zip(self.flag_values, self.flag_meanings.split(), strict=True))

    def check_dtype(self, dtype: np.dtype) -> None:
        """Refuse class values that `dtype` would store as other numbers.

        A class value names a class only as the exact number stored: 300 wraps
        to 44 on uint8, so it would name another class or none.

        Raises:
            ValueError: A class value or mask does not fit `dtype`.
        """
        values = [*(self.flag_values or ()), *(self.flag_masks or ())]
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(values).astype(dtype)
        misstored = [
            value for value, kept in zip(values, stored, strict=True) if kept != value
        ]
        if misstored:
            raise ValueError(
                f"The class values {misstored} do not fit this {dtype} band: they would "
                f"be stored as other numbers and label the wrong pixels. Use class "
                f"values the band can hold, or convert the band to a wider dtype first."
            )

    @model_validator(mode="before")
    @classmethod
    def _expand_classes(cls, data: Any) -> Any:
        """Turn the `classes` argument into the `flag_values` and `flag_meanings` CF stores."""
        if not isinstance(data, dict) or data.get("classes") is None:
            return data
        given = dict(data)
        classes = given.pop("classes")
        split = sorted(name for name in classes.values() if name.split() != [name])
        if split:
            raise ValueError(
                f"Class names {split} are empty or contain spaces, which the CF "
                f"flag_meanings format cannot store. Use single words such as "
                f"'tall_grass'."
            )
        values = sorted(classes)
        given["flag_values"] = values
        given["flag_meanings"] = " ".join(classes[value] for value in values)
        return given

    @model_validator(mode="after")
    def _check_names_match_values(self) -> Self:
        """Require one class name for each pixel value or bit mask listed."""
        names = [] if self.flag_meanings is None else self.flag_meanings.split()
        listings = {"pixel values": self.flag_values, "bit masks": self.flag_masks}
        listings = {label: listed for label, listed in listings.items() if listed}

        if listings and not names:
            raise ValueError(
                f"The flags lists {' and '.join(listings)} but names no classes. "
                f"Give each class a name."
            )
        if names and not listings:
            raise ValueError(
                f"The flags names the classes {names} but lists no pixel values. "
                f"Give each class the pixel value it stands for."
            )
        for label, listed in listings.items():
            if len(listed) != len(names):
                raise ValueError(
                    f"The flags lists {len(listed)} {label} but {len(names)} class "
                    f"names. Each class needs exactly one name and one of the {label}."
                )
        return self

    @model_validator(mode="after")
    def _check_values_ascending(self) -> Self:
        """Require pixel values in ascending order without repeats."""
        values = self.flag_values
        if (
            self.flag_masks is None
            and values is not None
            and values != sorted(set(values))
        ):
            raise ValueError(
                f"The flags's pixel values {values} must be ascending with no "
                f"repeats. Sort the classes by pixel value and remove duplicates."
            )
        return self
