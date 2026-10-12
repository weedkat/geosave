"""Cut windows in space into chips of one size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import pandas as pd
from affine import Affine
from odc.geo.geobox import GeoBox
from tiler import Tiler

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

# No "wrap": a chip reads only its own pixels, never the far edge a wrap needs.
type ChipMode = Literal["reflect", "edge", "constant"]


def layout(
    shape: tuple[int, int],
    size: tuple[int, int],
    *,
    overlap: int = 0,
    mode: ChipMode = "reflect",
) -> tuple[Tiler, list[tuple[int, int]]]:
    """Lay chips over a window, the one way both cutting and merging use.

    Overlapping chips are laid over a halo around the window, so every pixel
    of it sits away from a chip edge at least once and a tapered merge leaves
    no seam at the border.

    Args:
        shape: Height and width of the window.
        size: Height and width of one chip.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past the window is filled.

    Returns:
        The Tiler laying the chips, and the halo before and after each axis.
        The same arguments always give the same layout, so a merge rebuilds
        it instead of storing it.

    Examples:
        >>> tiler, halo = layout((300, 500), (128, 128), overlap=32)
        >>> len(tiler), halo
        (24, [(48, 48), (48, 48)])
    """
    tiler = Tiler(shape, size, overlap=overlap, mode=mode, constant_value=float("nan"))
    halo = [(0, 0), (0, 0)]
    if overlap:
        padded, halo = tiler.calculate_padding()
        tiler.recalculate(data_shape=padded)
    return tiler, [(int(before), int(after)) for before, after in halo]


def chips(
    windows: gpd.GeoDataFrame,
    size: int | tuple[int, int],
    *,
    overlap: int = 0,
    mode: ChipMode = "reflect",
) -> GeoDataFrame:
    """Cut each window into chips of one size.

    Args:
        windows: Windows as `stacks` or `frames` lists them.
        size: Chip side in pixels, or height and width.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past its window is filled.

    Returns:
        One row per chip, windows in the order given. `parent` is the
        position of the window cut, `chip` its number in that window's
        layout, `proj:transform`, `proj:shape` and `geometry` its own grid,
        `halo`, `overlap` and `mode` the layout a merge rebuilds, and `pad`
        the pixels of this chip past the window, filled as `mode` states.

    Examples:
        >>> cut = chips(stacks(items, on="label"), 224, overlap=32)
        >>> cut.iloc[0][["parent", "chip"]].astype(int).tolist(), cut.iloc[0]["proj:shape"]
        ([0, 0], [224, 224])
    """
    side = (size, size) if isinstance(size, int) else (size[0], size[1])
    cut = []
    for position, (_, window) in enumerate(windows.iterrows()):
        grid = window.gs.geobox
        height, width = grid.shape
        tiler, halo = layout((height, width), side, overlap=overlap, mode=mode)
        for number in range(len(tiler)):
            near, far = tiler.get_tile_bbox(number)
            row = int(near[0]) - halo[0][0]
            column = int(near[1]) - halo[1][0]
            h, w = (int(edge) for edge in far - near)
            chip = GeoBox(
                (h, w), grid.transform * Affine.translation(column, row), grid.crs
            )
            cut.append(
                {
                    **window.to_dict(),
                    "parent": position,
                    "chip": number,
                    "proj:transform": list(chip.transform)[:6],
                    "proj:shape": [h, w],
                    "halo": [list(widths) for widths in halo],
                    "overlap": overlap,
                    "mode": mode,
                    # This chip's pixels past the window, read by filling, not reading.
                    "pad": [
                        [max(0, -row), max(0, row + h - height)],
                        [max(0, -column), max(0, column + w - width)],
                    ],
                    "geometry": chip.extent.to_crs("EPSG:4326").geom,
                }
            )
    # No window cut leaves an empty table, which still names its columns.
    columns = pd.Index(
        dict.fromkeys([*windows.columns, "chip", "halo", "overlap", "mode", "pad"])
    )
    return cast(
        "GeoDataFrame",
        gpd.GeoDataFrame(
            cut, columns=columns, geometry="geometry", crs=windows.crs
        ).reset_index(drop=True),
    )
