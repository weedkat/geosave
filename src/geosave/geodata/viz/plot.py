"""Static figures behind `.gs.plot`, drawn through hvPlot and rendered with matplotlib.

`hv.save(figure, "scene.png")` writes one directly, and figures compose with
`+` (side by side) and `*` (overlaid). Class maps share a legend to the right
of the figure, with each name and colour shown once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import holoviews as hv
import hvplot.xarray  # noqa: F401  — registers .hvplot on xarray
import numpy as np
import spatialpandas as spd
from matplotlib.colors import to_hex, to_rgba

from geosave.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
    TIME_COORDINATE,
)
from geosave.geodata.viz import shapes
from geosave.geodata.viz.style import DEFAULT_CMAP, Stretch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import geopandas as gpd
    import xarray as xr
    from matplotlib.collections import PathCollection
    from holoviews.plotting.mpl.element import OverlayPlot

    from geosave.geodata.attrs import Legend

hv.extension("matplotlib")

# Width of lines in a figure, in points.
LINE_WIDTH = 2

# Element drawing each geometry type.
ELEMENTS: dict[str, type[hv.Element]] = {
    "Polygon": hv.Polygons,
    "LineString": hv.Path,
    "Point": hv.Points,
}


def draw_raster(
    raster: xr.Dataset,
    band: str | None = None,
    *,
    rgb: tuple[str, str, str] | None = None,
    cmap: str | Sequence[str] | None = None,
    clim: tuple[float, float] | None = None,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Draw one band of a raster, or a composite of three.

    A band draws through `cmap`, or its legend's class colours named beside the
    map; a composite draws red, green, blue stretched by `clim`. Values draw as
    stored, nodata included.

    Args:
        raster: Raster spanning the spatial pair and, optionally, `time`.
        band: Variable to draw. Exactly one of `band` and `rgb` is given.
        rgb: Variables drawn red, green and blue.
        cmap: Colormap of a band. Refused for classes and composites.
        clim: Colour limits of a band, or a composite's stretch. Refused for
            classes.
        cols: Columns a `time` axis lays panels into. Ignored otherwise.
        title: Panel title above the axes. None suppresses the title.
        xlabel: Caption appended below each panel's timestamp. None adds no
            caption.

    Returns:
        Image, RGB, or RGB with a legend over the raster's grid; or a Layout
        of one panel per `time` value, `cols` wide.

    Raises:
        ValueError: `cmap` accompanies classes or a composite, `clim`
            accompanies classes, a class carries no colour, or a composite's
            pixels are all nan or flat.

    Examples:
        >>> draw_raster(ds, "ndvi", cmap="YlGn")
        >>> draw_raster(ds, rgb=("B08", "B04", "B03"), clim=(0.0, 0.3))
    """
    y, x = SPATIAL_DIMENSIONS
    legend = None if band is None else raster[band].gs.legend
    has_classes = legend is not None and bool(legend.classes)

    # A band without classes draws through its colormap
    if band is not None and not has_classes:
        values = raster[band]
        plot = _draw_dates(
            values,
            lambda frame: frame.hvplot.image(
                x=x,
                y=y,
                cmap=cmap or DEFAULT_CMAP,
                clim=clim,
                colorbar=True,
                data_aspect=1,
                title="",
            ),
        )
        return _add_captions(plot, raster, cols=cols, title=title, xlabel=xlabel)

    # Classes and composites carry colour, not a quantity a colormap reads
    if cmap is not None:
        raise ValueError(
            "classes and composites draw their own colours, not a quantity; drop cmap"
        )
    if has_classes and clim is not None:
        raise ValueError(
            "the band lists classes, so clim cannot apply; its palette is "
            "Legend.color_map"
        )

    # Classes take their legend's colours; a composite stretches once for every date
    if has_classes:
        channels = raster[cast("str", band)].gs.colorize()
    else:
        channels = Stretch(clim).apply(raster[list(cast("tuple", rgb))].gs.to_array())
    plot = _draw_dates(
        channels,
        lambda frame: frame.hvplot.rgb(
            x=x, y=y, bands=BAND_DIMENSION, data_aspect=1, title=""
        ),
    )
    if has_classes:
        plot = plot * _draw_swatches(cast("Legend", legend))
        plot.opts(
            hv.opts.Overlay(
                legend_position="right", hooks=[_share_legend], backend="matplotlib"
            )
        )
    return _add_captions(plot, raster, cols=cols, title=title, xlabel=xlabel)


def draw_vector(frame: gpd.GeoDataFrame, *, color: str, alpha: float) -> hv.Overlay:
    """Draw every geometry of a frame in its own coordinates.

    Polygons, lines and points of one frame overlay each other.

    Args:
        frame: Frame with an active geometry column.
        color: Colour of outlines, lines and points; polygons fill with it at
            `alpha`.
        alpha: Opacity of the polygon fill.

    Returns:
        Overlay of one element per geometry type the frame holds.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    color = to_hex(color)
    options_by_element: dict[type[hv.Element], dict[str, object]] = {
        hv.Polygons: {"facecolor": to_rgba(color, alpha), "edgecolor": color},
        hv.Path: {"color": color, "linewidth": LINE_WIDTH},
        hv.Points: {"color": color},
    }
    # One element per geometry type, overlaid
    elements = []
    for geometry_type, rows in shapes.group_by_type(frame).items():
        element = ELEMENTS[geometry_type]
        elements.append(
            element(spd.GeoDataFrame(rows)).opts(
                **options_by_element[element], backend="matplotlib"
            )
        )
    plot = hv.Overlay(elements)
    plot.opts(aspect="equal", backend="matplotlib")
    return plot


def _draw_dates(
    array: xr.DataArray, draw: Callable[[xr.DataArray], hv.Element]
) -> hv.Element | hv.HoloMap:
    """Draw an array once, or once per date in a HoloMap keyed by `time`."""
    if TIME_COORDINATE not in array.dims:
        return draw(array)
    return hv.HoloMap(
        {
            time: draw(array.sel({TIME_COORDINATE: time}))
            for time in array[TIME_COORDINATE].values
        },
        kdims=[TIME_COORDINATE],
    )


def _draw_swatches(legend: Legend) -> hv.Overlay:
    """Return a legend entry per class: its name beside a square of its colour."""
    colors = legend.color_map or {}
    # A point off the axes carries each entry; the legend shows its marker
    return hv.Overlay(
        [
            hv.Points([(np.nan, np.nan)], label=name).opts(
                color=colors[code], marker="s", s=80, backend="matplotlib"
            )
            for code, name in (legend.classes or {}).items()
        ]
    )


def _share_legend(plot: OverlayPlot, element: hv.Overlay) -> None:
    """Move class swatches into one figure legend, including composed plots."""
    figure = plot.handles["fig"]
    legend = plot.handles["axis"].get_legend()
    if legend is None:
        return

    legends = [item for item in figure.legends if item.get_gid() == "geosave_classes"]
    entries: dict[tuple[str, tuple[float, ...]], PathCollection] = {}
    for item in [*legends, legend]:
        for handle, text in zip(item.legend_handles, item.get_texts(), strict=True):
            swatch = cast("PathCollection", handle)
            color = tuple(np.asarray(swatch.get_facecolor())[0])
            entries[(text.get_text(), color)] = swatch
        item.set_visible(False)
        item.remove()

    # Figure coordinates keep the legend beyond every panel, including layouts
    # assembled by the caller with `+`. HoloViews includes it in tight exports.
    shared = figure.legend(
        list(entries.values()),
        [name for name, color in entries],
        **{
            "prop": legend.prop,
            "ncols": plot.legend_cols or 1,
            **plot.legend_opts,
            "loc": "center left",
            "bbox_to_anchor": (1, 0.5),
            "bbox_transform": figure.transFigure,
        },
    )
    shared.set_gid("geosave_classes")
    plot.handles["bbox_extra_artists"].remove(legend)
    plot.handles["bbox_extra_artists"].append(shared)


def _add_captions(
    plot: hv.Element | hv.HoloMap,
    raster: xr.Dataset,
    *,
    cols: int,
    title: str | None,
    xlabel: str | None,
) -> hv.Element | hv.Layout:
    """Put panel metadata below the axes and caller titles above them."""

    def add_caption(panel: hv.Element, time_label: str | None) -> hv.Element:
        caption_lines = [] if time_label is None else [time_label]
        if xlabel:
            caption_lines.append(xlabel)
        options: dict[str, object] = {"show_title": title is not None}
        if title is not None:
            options["title"] = title
        if caption_lines:
            options["xlabel"] = "\n".join(caption_lines)
        return cast("hv.Element", panel.opts(**options, backend="matplotlib"))

    if isinstance(plot, hv.HoloMap):
        layout = plot.layout([TIME_COORDINATE])
        time_dimension = layout.kdims[0]
        # A flat Layout composes with `+`; an NdLayout cannot nest in one.
        panels = [
            add_caption(panel, str(time_dimension.pprint_value(time_label)))
            for time_label, panel in layout.items()
        ]
        return hv.Layout(panels).cols(cols)

    time_label = None
    if TIME_COORDINATE in raster.coords and raster[TIME_COORDINATE].ndim == 0:
        time_label = str(
            hv.Dimension(TIME_COORDINATE).pprint_value(raster[TIME_COORDINATE].values)
        )
    return add_caption(plot, time_label)
