"""The `gs` accessor of one cuts row: read the window it names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import orjson
import pandas as pd
import pystac
from affine import Affine
from odc.geo.geobox import GeoBox

if TYPE_CHECKING:
    from geosave.geodata import DataTree
    from geosave.geodata.transform.warp import Resampling


@pd.api.extensions.register_series_accessor("gs")
class GeoSample:
    """Read one row of `cuts.stacks`, `frames` or `chips` as a stack.

    Args:
        row: One row of a cuts table.

    Raises:
        AttributeError: The row states no `sources`, so it is no cuts row.

    Examples:
        >>> windows.iloc[0].gs.to_stack().gs.groups
        ('label', 's2')
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind the accessor to one row."""
        # pandas reads AttributeError as "this object has no such accessor".
        if "sources" not in row.index:
            raise AttributeError(
                "gs reads a row of a cuts table, which states its sources; "
                "cut one with cuts.stacks"
            )
        self._row = row

    @property
    def geobox(self) -> GeoBox:
        """Return the grid this row's window is read onto."""
        height, width = (int(edge) for edge in self._row["proj:shape"])
        transform = Affine(*self._row["proj:transform"][:6])
        return GeoBox(
            (height, width), transform, cast("str | None", self._row["proj:code"])
        )

    def to_stack(
        self, *, resampling: Mapping[str, Resampling] | None = None
    ) -> DataTree:
        """Read every collection of this window onto its grid, lazily.

        Args:
            resampling: Kernel per collection where its files are reprojected.
                A collection not named takes `"nearest"`, safe for classes.

        Returns:
            Stack with one group per collection, every group on this row's
            grid; pixels past the window it was cut from are filled as the
            row's `mode` states.

        Examples:
            >>> chip = windows.iloc[0].gs.to_stack(resampling={"s2": "bilinear"})
            >>> chip["s2"].dataset.sizes
            Frozen({'time': 4, 'y': 224, 'x': 224})
        """
        from geosave.geodata.core.stack import stack
        from geosave.geodata.io.raster import cogs
        from geosave.geodata.stac.item import band_attrs, cog_files, instant
        from geosave.geodata.transform.crop import window

        row = self._row
        grid = self.geobox
        height, width = grid.shape
        pad = row.get("pad")
        stated = [[0, 0], [0, 0]] if pad is None else [[int(a), int(b)] for a, b in pad]
        (top, bottom), (left, right) = stated
        inner = grid[top : height - bottom, left : width - right]

        kernels: dict[str, Resampling] = dict(resampling or {})
        groups = {}
        for name, members in orjson.loads(str(row["sources"])).items():
            # A frame lists an item once per instant it answers; read it once.
            unique = [
                pystac.Item.from_dict(item)
                for item in {item["id"]: item for item in members}.values()
            ]
            files = [file for item in unique for file in cog_files(item)]
            raster = cogs.read(
                files, geobox=inner, resampling=kernels.get(name, "nearest")
            )
            # The catalog's EO facts describe bands their files may not carry.
            stated = band_attrs(unique[0])
            raster = raster.assign(
                {
                    band: raster[band].assign_attrs(stated[band])
                    for band in raster.data_vars
                    if band in stated
                }
            )
            if len(unique) < len(members):
                raster = raster.sel(
                    time=pd.DatetimeIndex([instant(item) for item in members])
                )
            groups[name] = raster
        tree = stack(groups)
        if stated == [[0, 0], [0, 0]]:
            return tree
        return cast(
            "DataTree",
            window(
                tree,
                (-top, -left),
                (height, width),
                padding=[(top, bottom), (left, right)],
                mode=row.get("mode") or "constant",
            ),
        )
