"""What a stamp reads: the STAC side of one load."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
import pystac
import xarray as xr

from geosave.geodata.stac.item import read_instant


@dataclass(frozen=True)
class LoadSource:
    """The Items and Collection one load read.

    Attributes:
        items: Items that were loaded.
        collection: Collection the Items belong to, if known.
    """

    items: Sequence[pystac.Item]
    collection: pystac.Collection | None = None

    def match_steps(self, loaded: xr.Dataset) -> list[pystac.Item] | None:
        """Return the Item behind each `time` step, in step order.

        ODC labels each Item at its instant, so a step is one Item only when
        the Items' instants are distinct and the labels are exactly those.

        Args:
            loaded: Dataset the Items were loaded into.

        Returns:
            One Item per step; None for a timeless load or fused steps.
        """
        if "time" not in loaded.dims or loaded.sizes["time"] != len(self.items):
            return None

        by_instant = {}
        for item in self.items:
            instant = read_instant(item)
            if instant is None or instant in by_instant:
                return None
            by_instant[instant] = item

        ordered = []
        for label in pd.DatetimeIndex(loaded.time.values):
            if label not in by_instant:
                return None
            ordered.append(by_instant[label])
        return ordered


type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]
"""Write what the STAC documents state onto a loaded Dataset, reading no pixels."""
