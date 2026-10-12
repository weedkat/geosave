"""What a whole dataset is, as ACDD discovery attributes on its root."""

from __future__ import annotations

from typing import ClassVar

from .base import AttrsGroup, Join


class ACDD(AttrsGroup):
    """Describe a whole dataset with the ACDD keys catalogs search on.

    It lives on a Dataset's root attrs, where a STAC Collection's fields land
    when a source loads it.

    Args:
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

    # Each key is a fact on its own: a shared license outlives differing titles.
    JOIN: ClassVar[Join] = "each"

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    license: str | None = None
