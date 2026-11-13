"""Items and loaded stand-ins for stamp tests: no pixels, no network."""

from datetime import UTC, datetime

import numpy as np
import pystac
import xarray as xr


def scene(id: str = "scene", day: int = 1, **properties) -> pystac.Item:
    """Return an Item acquired on 2025-01-`day` with the given properties."""
    return pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=datetime(2025, 1, day, tzinfo=UTC),
        properties=properties,
    )


def with_asset(
    item: pystac.Item, key: str, *, title: str | None = None, **fields
) -> pystac.Item:
    """Add one asset stating `fields`, such as `eo:bands`, to `item`."""
    item.add_asset(
        key,
        pystac.Asset(
            href=f"https://example.com/{item.id}/{key}.tif",
            media_type=pystac.MediaType.GEOTIFF,
            title=title,
            extra_fields=fields,
        ),
    )
    return item


def pixels(*names: str) -> xr.Dataset:
    """Stand in for an ODC result: one 1x1 variable per name, without attrs."""
    variables = {}
    for name in names:
        variables[name] = (("y", "x"), [[1]])
    return xr.Dataset(variables)


def stepped(*items: pystac.Item) -> xr.Dataset:
    """Stand in for an ODC result with one `time` step per Item, in this order."""
    times = []
    for item in items:
        times.append(item.datetime.replace(tzinfo=None))
    return xr.Dataset(
        {"red": (("time", "y", "x"), np.zeros((len(items), 1, 1)))},
        coords={"time": np.array(times, dtype="datetime64[ns]")},
    )
