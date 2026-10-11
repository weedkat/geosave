"""Merge chip outputs saved during prediction back into whole rasters."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
from tiler import Merger

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS
from geosave_engine.geodata.core.constructors import raster

from .chips import layout

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset

# How overlapping chips are weighed, as the window `tiler` names.
type Taper = Literal[
    "boxcar",
    "hamming",
    "triang",
    "bartlett",
    "hann",
    "barthann",
    "bohman",
    "parzen",
    "blackman",
    "nuttall",
    "blackmanharris",
    "overlap-tile",
]


def merge(
    parents: gpd.GeoDataFrame,
    chips: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    *,
    taper: Taper | None = None,
) -> dict[int, Dataset]:
    """Merge saved chip outputs onto the grid of each window they were cut from.

    The layout a cut used is rebuilt from the parent's shape and the chips'
    settings, so chips may have been predicted in any order and by several
    writers. Overlaps are weighed and summed, as logits are; classes are
    taken after merging, with `argmax`.

    Args:
        parents: The windows the chips were cut from, as given to `chips`.
        chips: The chips, as `chips` returned them.
        predictions: Item table of the saved outputs, one per chip, with a
            `chip` column holding the chip's row position.
        taper: How overlapping chips are weighed, as a window name such as
            `"hann"`. None weighs every chip alike.

    Returns:
        {
            <parent position>: Dataset on the parent's grid, holding the
                merged output as `(band, y, x)` under the bands' shared name,
                or `"output"`,
        }

    Raises:
        ValueError: A parent's chip has no saved output, or `taper` is given
            for chips cut without overlap.

    Examples:
        >>> merged = merge(frames, chips, ChipWriter.read("runs/pred"), taper="hann")
        >>> merged[0]["logits"].argmax("band")
        <xarray.DataArray 'logits' (y: 512, x: 512)> ...
    """
    from geosave_engine.geodata.io.readers import open_raster
    from geosave_engine.geodata.stac._item import cog_files
    from geosave_engine.geodata.io.vector.stac_geoparquet import to_items

    outputs = {
        int(position): cog_files(item.to_dict())
        for position, item in zip(
            predictions["chip"], to_items(predictions), strict=True
        )
    }
    merged: dict[int, Dataset] = {}
    # A tapered edge weighs nearly nothing, so only an overlapping chip restores it.
    if taper is not None and (chips["overlap"] == 0).any():
        raise ValueError(
            f"a {taper!r} taper fades every chip edge, and chips cut without "
            f"overlap leave no neighbour to cover it; cut chips with overlap or "
            f"pass taper=None"
        )
    for key, members in chips.groupby("parent", sort=True):
        # pandas types a group key as Hashable; `parent` holds window positions.
        parent = int(cast("int", key))
        absent = [int(p) for p in members.index if int(p) not in outputs]
        if absent:
            raise ValueError(
                f"window {parent} cannot be merged: chips {absent} have no "
                f"output; predict them first"
            )
        grid = parents.iloc[parent].gs.geobox
        height, width = grid.shape
        first = members.iloc[0]
        chip_height, chip_width = (int(edge) for edge in first["proj:shape"])
        tiler, halo = layout(
            (height, width),
            (chip_height, chip_width),
            overlap=int(first["overlap"]),
            mode=first["mode"],
        )
        merger = None
        name = "output"
        for position, number in zip(members.index, members["chip"], strict=True):
            output = open_raster(outputs[int(position)])
            bands = list(output.gs.variables)
            name = os.path.commonprefix(bands).rstrip("_") or "output"
            values = np.stack([output[band].values.squeeze() for band in bands])
            if merger is None:
                merger = Merger(
                    tiler, logits=len(bands), window=taper, save_visits=False
                )
            merger.add(int(number), values)
        assert merger is not None
        data = merger.merge(extra_padding=halo)
        merged[parent] = raster({name: (("band", *SPATIAL_DIMENSIONS), data)}, grid)
    return merged
