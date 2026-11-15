"""What a stamp reads: the STAC side of one load."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pystac
import xarray as xr
from odc.stac.model import PropertyLoadRequest


@dataclass(frozen=True)
class LoadSource:
    """The Items and property requests of one load.

    Attributes:
        items: Items that were loaded; each links to its Collection.
        properties: The `with_properties` requests, as ODC normalized them.
    """

    items: Sequence[pystac.Item]
    properties: Sequence[PropertyLoadRequest] = ()


type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]
"""Write what the STAC documents state onto a loaded Dataset, reading no pixels."""
