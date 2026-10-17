"""Per-variable pixel summaries over one or many rasters."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.base import compute


def statistics(rasters: Iterable[Mapping[Hashable, xr.DataArray]]) -> pd.DataFrame:
    """Summarise each variable over the present pixels of every raster.

    Pixels are pooled across rasters, so rasters may differ in grid and size;
    each variable reduces across all its axes, excluding NaN and the fill
    value each raster declares. Every raster is read in one dask compute,
    whose reductions release each chunk once it is summarised. Register
    `dask.diagnostics.ProgressBar` to watch a long one.

    Args:
        rasters: Datasets sharing the same variables, such as training tiles.

    Returns:
        One row per variable, in the first raster's order, with minimum,
        maximum, mean, population stddev, and valid_percent columns. No
        rasters gives an empty table with these columns.

    Raises:
        ValueError: A raster's variables differ from the first raster's, or
            a variable holds no present pixel.

    Examples:
        >>> tiles = [gs.open_raster(path) for path in paths]
        >>> gs.statistics(tiles)
             minimum  maximum  mean  stddev  valid_percent
        B04   0.0001   1.6528  0.12    0.08          100.0
        B08   0.0001   1.5872  0.25    0.08           99.9
    """
    rasters = list(rasters)
    names = list(rasters[0]) if rasters else []
    for raster in rasters[1:]:
        if list(raster) != names:
            raise ValueError(
                f"statistics pool variables by name, but a raster holds "
                f"{list(raster)} where the first holds {names}"
            )

    columns = ["minimum", "maximum", "mean", "stddev", "valid_percent"]
    if not names:
        return pd.DataFrame(columns=pd.Index(columns))

    # One block per raster holds all its variables, so a source band several
    # variables read, as NDVI reads red, is released with its raster rather
    # than waiting for every raster's later variables.
    blocks = []
    sizes = np.zeros(len(names))
    for raster in rasters:
        rows = []
        for position, name in enumerate(names):
            band = raster[name]
            fill = band.gs.nodata
            kept = band if fill is None else band.where(band != fill)
            rows.append(da.asarray(kept.data, dtype="float64").ravel())
            sizes[position] += band.size
        # Variables spanning other axes are padded with NaN, which counts as absent.
        width = max(row.size for row in rows)
        padded = []
        for row in rows:
            if row.size < width:
                row = da.concatenate([row, da.full(width - row.size, np.nan)])
            padded.append(row)
        blocks.append(da.stack(padded).rechunk({0: -1}))

    pixels = xr.DataArray(da.concatenate(blocks, axis=1), dims=("variable", "pixel"))
    present = pixels.notnull().sum("pixel")
    summary = xr.Dataset(
        {
            "present": present,
            "minimum": pixels.min("pixel"),
            "maximum": pixels.max("pixel"),
            "mean": pixels.mean("pixel"),
            "stddev": pixels.std("pixel"),
            "valid_percent": 100.0 * present / sizes,
        }
    )
    (summary,) = compute(summary)

    for name, count in zip(names, summary["present"].values, strict=True):
        if not count:
            raise ValueError(
                f"{name} holds no present pixel, so it summarises to "
                f"nothing; drop the band or give it pixels that are not fill"
            )
    table = summary[columns].to_dataframe()
    table.index = pd.Index(names)
    return table


def class_percent(bands: Iterable[xr.DataArray]) -> pd.Series:
    """Give the share of present pixels in each class, pooled over bands.

    Bands may differ in grid and size but must share one legend. Pixels equal
    to a band's fill value are not counted. Every band is read in one dask
    compute.

    Args:
        bands: Categorical bands sharing the same legend, such as the label of
            each training tile.

    Returns:
        Percent per class, named `percent` and indexed by class name in legend
        order; classes that never occur read 0.

    Raises:
        ValueError: No bands are given, a band states no class names, or the
            legends differ.

    Examples:
        >>> labels = [gs.open_raster(path)["label"] for path in paths]
        >>> gs.class_percent(labels)
        water    12.5
        trees    60.0
        crops    27.5
        Name: percent, dtype: float64
    """
    bands = list(bands)
    if not bands:
        raise ValueError("class_percent needs at least one band to count")
    classes = bands[0].gs.legend.classes if bands[0].gs.legend else None
    if classes is None:
        raise ValueError(
            "the band states no class names, so its values name no classes; "
            "write a legend with write_legend"
        )

    counts = []
    for band in bands:
        legend = band.gs.legend
        if legend is None or legend.classes != classes:
            raise ValueError(
                f"class_percent pool bands by class, but one band's legend "
                f"differs from {classes}"
            )
        fill = band.gs.nodata
        present = band if fill is None else band.where(band != fill)
        # A fill class such as "unlabelled" is not a class of present pixels.
        counted = [code for code in classes if code != fill]
        counts.append(
            xr.concat([(present == code).sum() for code in counted], dim="class")
        )
    (total,) = compute(sum(counts))

    names = [name for code, name in classes.items() if code != fill]
    percent = 100.0 * total.values / total.values.sum()
    return pd.Series(percent, index=names, name="percent")
