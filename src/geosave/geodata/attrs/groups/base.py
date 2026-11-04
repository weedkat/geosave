"""What an attr group is: a few attr keys that mean one value together.

Each group is a pydantic model over its keys; `from_attrs` reads one off an
object's attrs and `attrs.stamp_attrs` writes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Self

import numpy as np
import orjson
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
)
from xarray.core.utils import equivalent


def decode(value: object) -> object:
    """Read GDAL text and NumPy scalars as native Python values.

    Args:
        value: Attribute value, optionally JSON text or a NumPy scalar.

    Returns:
        Parsed JSON value, a NumPy scalar's Python value, or the original
        value when neither conversion applies. Non-JSON strings stay strings.
    """
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

    A group only knows its own keys and how to read and write itself. Joining
    many objects' attrs, and what to do on a conflict, belongs to
    `attrs.combine`, which keeps a group whole or drops it.

    Attributes:
        KEYS: Every key the group reads; stamping replaces all of them.
    """

    # Groups are values; the other keys of an attrs mapping are none of theirs.
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    KEYS: ClassVar[tuple[str, ...]] = ()

    # Pydantic calls this once a subclass's fields exist; `__init_subclass__`
    # runs too early to see them. A field may be stored under an alias, such as
    # a CF or GDAL attr name, or under several (`AliasChoices`), so `KEYS` is
    # the attr names the group reads, not the Python field names.
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
    def is_stated_in(cls, attrs: Mapping[Any, Any]) -> bool:
        """Say whether the attrs state any key of the group.

        Args:
            attrs: Attribute mapping to inspect, including accepted aliases.

        Returns:
            True when at least one group key is present, even if its value
            would fail validation. No values are parsed.
        """
        return any(key in attrs for key in cls.KEYS)

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self | None:
        """Read the group from attrs, None where they state none of its keys.

        Args:
            attrs: Attribute mapping containing group fields or their aliases.
                Unrelated keys are ignored.

        Returns:
            Validated group, or None when no group key is present.

        Raises:
            ValueError: The attrs state the group invalidly.
        """
        return cls.model_validate(dict(attrs)) if cls.is_stated_in(attrs) else None

    def to_attrs(self) -> dict[str, Any]:
        """Return the attrs this value writes; a field at its default or None writes none.

        netCDF refuses a None attr, so an unset field leaves its key out.

        Returns:
            New mapping using serialization aliases and JSON-compatible
            values, omitting fields at their defaults or set to None.
        """
        return self.model_dump(
            by_alias=True, mode="json", exclude_defaults=True, exclude_none=True
        )

    # Defining `__eq__` makes a group unhashable on purpose: NaN equals NaN here
    # but hashes by identity, and a dict-valued field cannot hash at all.
    def __eq__(self, other: object) -> bool:
        """Compare the attrs each group writes, treating NaN as equal to NaN.

        Args:
            other: Value to compare with this group.

        Returns:
            True when both groups have the same concrete type and serialize
            equivalent attrs; False when those attrs differ. Returns
            NotImplemented for another type so Python can try its comparison.
        """
        if type(other) is not type(self):
            return NotImplemented
        mine, theirs = self.to_attrs(), other.to_attrs()
        return mine.keys() == theirs.keys() and all(
            equivalent(mine[key], theirs[key]) for key in mine
        )
