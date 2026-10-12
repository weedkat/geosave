"""What an attr group is, and how one is parsed off a band and stamped onto it.

A group is a few attr keys that mean one value together, validated by a
pydantic model. `parse` reads it lazily from a band's attrs; `stamp` writes it,
replacing every key the group reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import numpy as np
import orjson
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
)
from xarray.core.utils import equivalent

if TYPE_CHECKING:
    import xarray as xr

type Join = Literal["agree", "drop"]


def decode(value: object) -> object:
    """Read GDAL's text spelling and NumPy scalars as the native values they spell."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError:
            return value
    return value


# GDAL tags are text; Zarr, NetCDF and memory are native. Both read the same.
type Decoded[T] = Annotated[T, BeforeValidator(decode)]


class AttrsGroup(BaseModel):
    """A set of attr keys read and written as one validated value.

    A subclass is joined by `combine` as soon as it is defined. Its keys are
    its fields' names, aliases, or alias choices.

    Attributes:
        KEYS: Every key the group reads; stamping replaces all of them.
        JOIN: What a join does when objects state different values:
            `"agree"` refuses it, `"drop"` leaves the group out.
    """

    # Groups are values; the other keys of an attrs mapping are none of theirs.
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    KEYS: ClassVar[tuple[str, ...]] = ()
    JOIN: ClassVar[Join]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        keys: list[str] = []
        for name, field in cls.model_fields.items():
            if isinstance(field.validation_alias, AliasChoices):
                keys += [str(choice) for choice in field.validation_alias.choices]
            else:
                keys.append(field.alias or name)
        cls.KEYS = tuple(keys)

    @classmethod
    def stated_in(cls, attrs: Mapping[Any, Any]) -> bool:
        """Say whether the attrs state any key of the group."""
        return any(key in attrs for key in cls.KEYS)

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self | None:
        """Read the group from attrs, None where they state none of its keys.

        Raises:
            ValueError: The attrs state the group invalidly.
        """
        return cls.model_validate(dict(attrs)) if cls.stated_in(attrs) else None

    def held_by(self, band: xr.DataArray) -> Self:
        """Return the value as `band` stores it; most groups name no pixel value.

        Raises:
            ValueError: The band's dtype cannot hold the value.
        """
        return self

    def to_attrs(self) -> dict[str, Any]:
        """Return the attrs this value writes; a field at its default or None writes none.

        netCDF refuses a None attr, so an unset field leaves its key out.
        """
        return self.model_dump(
            by_alias=True, mode="json", exclude_defaults=True, exclude_none=True
        )

    def __eq__(self, other: object) -> bool:
        """Compare by the attrs each writes, as xarray compares attrs: NaN equals NaN."""
        if type(other) is not type(self):
            return NotImplemented
        mine, theirs = self.to_attrs(), other.to_attrs()
        return mine.keys() == theirs.keys() and all(
            equivalent(mine[key], theirs[key]) for key in mine
        )

    __hash__ = BaseModel.__hash__


def parse[G: AttrsGroup](band: xr.DataArray, group: type[G]) -> G | None:
    """Read one group off a band, validated against the band's dtype.

    Args:
        band: Variable whose attrs state the group.
        group: Group to read.

    Returns:
        The group's value, or None where the band states none of its keys.

    Raises:
        ValueError: The attrs state the group invalidly or the band's dtype
            cannot hold it.

    Examples:
        >>> parse(ds["B04"], Nodata)
        Nodata(value=0)
    """
    value = group.from_attrs(band.attrs)
    return None if value is None else value.held_by(band)


def stamp[G: AttrsGroup, X: xr.DataArray](
    band: X, group: type[G], value: G | None
) -> X:
    """Return a copy of the band stating `value` for one group.

    Every key of the group is removed first, so no stale spelling survives.

    Args:
        band: Variable to stamp.
        group: Group the value belongs to.
        value: Value to state. None removes the group.

    Returns:
        Shallow copy of the band with replaced attrs; pixels are not copied.

    Raises:
        ValueError: The band's dtype cannot hold the value.

    Examples:
        >>> stamp(ds["B04"], Nodata, Nodata(value=0)).attrs["_FillValue"]
        0
    """
    attrs = {key: v for key, v in band.attrs.items() if key not in group.KEYS}
    if value is not None:
        attrs |= value.held_by(band).to_attrs()
    stamped = band.copy(deep=False)
    stamped.attrs = attrs
    return stamped
