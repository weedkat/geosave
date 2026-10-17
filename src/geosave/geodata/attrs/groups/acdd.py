"""What a whole dataset is, as ACDD discovery attributes on its root."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from .base import AttrsGroup


class ACDD(AttrsGroup):
    """Describe a whole dataset with the ACDD keys catalogs search on.

    It lives on a Dataset's root attrs, where a STAC Collection's fields land
    when a source loads it.

    Attributes:
        id: Identifier of the dataset, a STAC Collection's id.
        title: Short human-readable name.
        summary: Paragraph describing the data, a Collection's description.
        keywords: Comma-separated keywords.
        institution: Organization that produced the data.
        license: License of the data, an SPDX identifier where one applies.

    Examples:
        >>> ds.gs.acdd.title
        'Sentinel-2 Level-2A'
    """

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    license: str | None = None

    @classmethod
    def intersect_values(cls, values: Sequence[Self | None]) -> Self | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # Self narrows in a classmethod
        """Keep each field every object states alike.

        Each field is a fact on its own, so a shared license outlives
        differing titles.

        Args:
            values: Each object's ACDD, None where it states none.

        Returns:
            The fields all objects share, or None where any object states none.
        """
        if None in values:
            return None
        first, *others = values
        alike = {
            name: getattr(first, name)
            for name in cls.model_fields
            if all(getattr(other, name) == getattr(first, name) for other in others)
        }
        return cls(**alike)
