"""What a stamp reads: the STAC side of one load."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pystac
import xarray as xr


@dataclass(frozen=True)
class LoadSource:
    """The Items and Collection one load read.

    Attributes:
        items: Items that were loaded.
        collection: Collection the Items belong to, if known.
    """

    items: Sequence[pystac.Item]
    collection: pystac.Collection | None = None


type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]
"""Write what the STAC documents state onto a loaded Dataset, reading no pixels."""
