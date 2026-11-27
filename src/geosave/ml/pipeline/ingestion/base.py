"""Contract for a reader of one raster collection, and the ingest stage's output."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from geosave.geodata import GeoRegion


@dataclass(frozen=True)
class IngestOutput:
    """What the ingest stage hands the preprocessor: every collection in one tree.

    Attributes:
        sources: One child per collection, in memory or reopened lazily from
            the run's store.
        metadata: Values a custom stage adds for a later one.
    """

    sources: xr.DataTree
    metadata: dict[str, Any] = field(default_factory=dict)


class Ingestor(ABC):
    """A reader of one collection's rasters over a region.

    Each ingestor returns its own subtree under its name, lazily, so a
    pipeline puts several side by side under one root and decides whether
    to load them into memory or stage them.
    """

    @abstractmethod
    def __call__(self, region: GeoRegion) -> xr.DataTree:
        """Read the collection over a region, lazily.

        Args:
            region: Place and time span to read.
        """
