"""Samples of an Item table, for training: being redesigned."""

from __future__ import annotations

from typing import Any


class GeoSamples:
    """Samples of catalog Items, planned for training.

    Raises:
        NotImplementedError: Always, while sampling is redesigned to take a
            STAC DataTree.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Refuse construction until sampling is redesigned."""
        raise NotImplementedError(
            "GeoSamples is being redesigned to take a STAC DataTree"
        )
