"""Merge chip outputs back into whole rasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
from tiler import Merger

from geosave.geodata.conventions import SPATIAL_DIMENSIONS
from geosave.geodata.core.constructors import raster

from .chips import layout

if TYPE_CHECKING:
    from geosave.geodata import Dataset

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
    outputs: Mapping[int, np.ndarray],
    *,
    bands: Sequence[str],
    valid: Mapping[int, np.ndarray] | None = None,
    taper: Taper | None = None,
) -> dict[int, Dataset]:
    """Merge chip outputs onto the grid of each window they were cut from.

    The layout a cut used is rebuilt from the parent's shape and the chips'
    settings, so outputs may arrive in any order. Each pixel is the weighted
    mean of the chips covering it validly, weighed by `taper`; a pixel no
    chip covers validly is NaN. Overlaps are averaged as logits are; take
    classes after merging.

    Args:
        parents: The windows the chips were cut from, as given to `chips`.
        chips: The chips, as `chips` returned them, or a subset of them.
        outputs: Each chip's row label in `chips` mapped to its
            `(band, y, x)` output.
        bands: Name of each output band, in order.
        valid: Each chip's row label mapped to its `(y, x)` pixels that
            count. None counts every finite pixel.
        taper: How overlapping chips are weighed, as a window name such as
            `"hann"`. None weighs every chip alike.

    Returns:
        {
            <parent position>: float64 Dataset on the parent's grid, one
                `(y, x)` variable per band,
        }

    Raises:
        ValueError: A chip has no output, an output holds other than one
            band per name, or `taper` is given for chips cut without overlap.

    Examples:
        >>> merged = merge(frames, chips, outputs, bands=["background", "palm"])
        >>> merged[0].to_dataarray("band").argmax("band")
        <xarray.DataArray (y: 512, x: 512)> ...
    """
    # A tapered edge weighs nearly nothing, so only an overlapping chip restores it.
    if taper is not None and (chips["overlap"] == 0).any():
        raise ValueError(
            f"a {taper!r} taper fades every chip edge, and chips cut without "
            f"overlap leave no neighbour to cover it; cut chips with overlap or "
            f"pass taper=None"
        )
    merged: dict[int, Dataset] = {}
    for key, members in chips.groupby("parent", sort=True):
        # pandas types a group key as Hashable; `parent` holds window positions.
        parent = int(cast("int", key))
        absent = [int(label) for label in members.index if int(label) not in outputs]
        if absent:
            raise ValueError(
                f"window {parent} cannot be merged: chips {absent} have no "
                f"output; predict them first"
            )
        grid = parents.iloc[parent].gs.geobox
        first = members.iloc[0]
        tiler, halo = layout(
            grid.shape,
            (int(first["proj:shape"][0]), int(first["proj:shape"][1])),
            overlap=int(first["overlap"]),
            mode=first["mode"],
        )
        # A last channel carries each chip's validity, tapered as its values are.
        merger = Merger(
            tiler,
            logits=len(bands) + 1,
            window=taper,
            save_visits=False,
            data_dtype=np.float64,
            weights_dtype=np.float64,
        )
        for label, number in zip(members.index, members["chip"], strict=True):
            values = np.asarray(outputs[int(label)], dtype=np.float64)
            if len(values) != len(bands):
                raise ValueError(
                    f"chip {int(label)} holds {len(values)} bands, but bands "
                    f"names {len(bands)}: {list(bands)}"
                )
            counts = np.isfinite(values).all(axis=0)
            if valid is not None:
                counts &= np.asarray(valid[int(label)], dtype=bool)
            contribution = np.concatenate([np.where(counts, values, 0), counts[None]])
            merger.add(int(number), contribution)
        sums = merger.merge(extra_padding=halo, normalize_by_weights=False)
        coverage = sums[-1]
        means = np.full(sums[:-1].shape, np.nan)
        np.divide(sums[:-1], coverage, out=means, where=coverage > 0)
        merged[parent] = raster(
            {
                name: (SPATIAL_DIMENSIONS, band)
                for name, band in zip(bands, means, strict=True)
            },
            grid,
        )
    return merged
