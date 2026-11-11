"""Static raster figures: one band through a colormap or its classes, or a composite."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import holoviews as hv
import hvplot.xarray  # noqa: F401  — registers .hvplot on xarray
import numpy as np

from geosave.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
    TIME_COORDINATE,
)
from geosave.geodata.viz.colors import DEFAULT_CMAP, stretch_channels

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import xarray as xr
    from matplotlib.collections import PathCollection
    from holoviews.plotting.mpl.element import OverlayPlot

    from geosave.geodata.attrs import Legend


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
    stored, nodata included. Naming neither draws the sole variable, else the
    composite `rgb` records.

    Args:
        raster: Raster spanning the spatial pair and, optionally, `time`.
        band: Variable to draw.
        rgb: Variables drawn red, green and blue.
        cmap: Colormap of a band. Refused for classes and composites.
        clim: Colour limits of a band, or a composite's stretch. Refused for
            classes.
        cols: Columns a `time` axis lays panels into. Ignored otherwise.
        title: Panel title above the axes. None suppresses the title.
        xlabel: Caption appended below each panel's timestamp. None uses the
            raster's reverse-geocoded location; an empty string suppresses it.

    Returns:
        Image, RGB, or RGB with a legend over the raster's grid; or a Layout
        of one panel per `time` value, `cols` wide.

    Raises:
        KeyError: A named variable is absent.
        ValueError: Both `band` and `rgb` are named, neither is while the
            raster carries several variables and records no composite, `cmap`
            accompanies classes or a composite, `clim` accompanies classes, or
            a composite's pixels are all nan or flat.

    Examples:
        >>> draw_raster(ds, "ndvi", cmap="YlGn")
        >>> draw_raster(ds, rgb=("B08", "B04", "B03"), clim=(0.0, 0.3))
    """
    band, rgb = _choose_variables(raster, band, rgb)
    if xlabel is None:
        place = raster.gs.region.locate()
        xlabel = place.to_address() if place is not None else None

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
        channels = stretch_channels(
            raster[list(cast("tuple", rgb))].gs.to_dataarray(), clim
        )
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


def _choose_variables(
    raster: xr.Dataset, band: str | None, rgb: tuple[str, str, str] | None
) -> tuple[str | None, tuple[str, str, str] | None]:
    """Return the band or composite to draw: the one named, the sole variable, or the recorded composite.

    Raises:
        KeyError: A named variable is absent.
        ValueError: Both are named, or neither while the raster carries several
            variables and records no composite.
    """
    if band is not None and rgb is not None:
        raise ValueError("name a band or a composite, not both")
    if band is None and rgb is None:
        variables = list(raster.gs.raster_variables)
        if len(variables) == 1:
            band = variables[0]
        else:
            rgb = raster.gs.rgb
        if band is None and rgb is None:
            raise ValueError(
                f"this raster holds {variables} and records no composite; name "
                f"one band, ds.gs.plot('ndvi'), or a composite, "
                f"ds.gs.plot(rgb=('B04', 'B03', 'B02')), or record one with "
                f"ds.gs.write_rgb('B04', 'B03', 'B02')"
            )
    drawn = [band] if band is not None else list(cast("tuple", rgb))
    missing = [name for name in drawn if name not in raster.data_vars]
    if missing:
        raise KeyError(
            f"{missing} are not data variables of this raster; it carries "
            f"{list(raster.gs.raster_variables)}"
        )
    return band, rgb


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
    colors = legend.colors
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

    # One entry per name and colour, from this panel's legend and any shared one
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
