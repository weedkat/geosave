"""Raster display choices, and the cached pixels of the map's view."""

from __future__ import annotations

from typing import cast

import geoviews as gv
from bokeh.models import HoverTool
import holoviews as hv
import numpy as np
import pandas as pd
import param
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata.attrs import stamp_attrs
from geosave.geodata.transform.color import colorize
from geosave.geodata.viz.colors import (
    CMAPS,
    DEFAULT_CMAP,
    find_percentile_range,
    stretch_channels,
)

from ..mercator import (
    DEFAULT_VIEW_SHAPE,
    create_view,
    create_zoom_grid,
    find_zoom_level,
    to_web_mercator,
)
from .layer import Layer

# Largest side, in pixels, of the scene overview a composite's contrast fits.
OVERVIEW_SIZE = 256


def _create_rgb(x, y, red, green, blue, shown) -> gv.RGB:
    """Return an RGB image on the map, transparent where `shown` is 0."""
    return gv.RGB(
        (x, y, red, green, blue, shown),
        crs=gv.tile_sources.OSM.crs,
        vdims=["R", "G", "B", "A"],
    )


def _fit_limits(values: np.ndarray, percentiles) -> tuple[float, float]:
    """Return the values at `percentiles` of the finite values, never empty.

    Unlike `find_stretch_limits`, a map keeps drawing: no values fit (0, 1),
    and flat values a range one wide.
    """
    limits = find_percentile_range(values, percentiles)
    if limits is None:
        return 0.0, 1.0
    low, high = limits
    return low, high if high > low else low + 1


class RasterLayer(Layer):
    """One raster on the map: a band through a colormap or its classes, or a composite.

    Display choices never change the raster's pixels or metadata.
    """

    heading = "Rasters"
    icon = "grid_on"

    draw_params = (
        *Layer.draw_params,
        "band",
        "rgb",
        "cmap",
        "clim",
        "stretch_percentiles",
        "time",
        "axis_positions",
    )
    legend_params = ("band", "rgb", "cmap", "clim", "fitted_clim")

    cache_bytes = param.Integer(
        128 * 1024**2,
        bounds=(0, None),
        precedence=-1,
        doc="Byte budget for reusable viewport pixel windows.",
    )

    band = param.Selector(default=None, allow_None=True, label="Display")
    rgb = param.Tuple(default=None, length=3, allow_None=True, precedence=-1)
    cmap = param.Selector(
        default=DEFAULT_CMAP, objects=CMAPS, check_on_set=False, label="Colormap"
    )
    stretch_percentiles = param.Range(
        (2.0, 98.0), bounds=(0.0, 100.0), label="Stretch percentiles"
    )
    clim = param.Range(None, allow_None=True, label="Color limits")
    fitted_clim = param.Range((0.0, 1.0), precedence=-1)
    time = param.Selector(default=None, allow_None=True, label="Date")
    axis_positions = param.Dict(default={}, precedence=-1)

    def __init__(
        self,
        raster,
        band=None,
        *,
        row: pd.Series | None = None,
        rgb=None,
        name=None,
        cmap=None,
        alpha=0.8,
    ):
        """Show one band, or a composite of three, of a raster with a grid.

        Args:
            raster: Dataset, or a DataArray with or without a `band` axis.
            band: Band shown first. None shows the declared composite, else
                the first band.
            row: Catalog row whose files hold `raster`'s pixels, read from
                their overviews. None computes the pixels from `raster`.
            rgb: Bands composited red, green and blue, overriding the
                raster's declared composite.
            name: Name in the layer list. None names it after what it shows.
            cmap: Colormap of a continuous band.
            alpha: Opacity over the basemap.

        Raises:
            ValueError: The raster has no grid or spatial band, its longitudes
                run past 180°, or both `band` and `rgb` are given.
            KeyError: `band` or an `rgb` channel is no spatial band.
        """
        # 1. A Dataset with a grid a web map can place
        if isinstance(raster, xr.DataArray):
            raster = (
                raster.gs.to_raster()
                if "band" in raster.dims
                else raster.to_dataset(name=raster.name or "band")
            )
        self._check_mappable(raster)
        if band is not None and rgb is not None:
            raise ValueError("choose a band or RGB, not both")
        self.data = raster
        self.row = row

        # 2. The spatial bands, and the band and composite chosen among them
        self.bands = []
        for band_name, variable in raster.data_vars.items():
            if {"y", "x"} <= set(variable.dims):
                self.bands.append(str(band_name))
        if not self.bands:
            raise ValueError("the raster has no spatial bands")
        for choice in [band, *(rgb or ())]:
            if choice is not None and choice not in self.bands:
                raise KeyError(f"{choice!r} is not a spatial band")
        channels = rgb or raster.gs.rgb
        if channels is not None and any(name not in self.bands for name in channels):
            raise KeyError("the declared RGB includes a non-spatial band")
        super().__init__(
            name=name or str(band or ("RGB" if channels else self.bands[0])),
            alpha=alpha,
            cmap=cmap or DEFAULT_CMAP,
        )

        # 3. What the sidebar offers: displays, colormaps, dates, other axes
        self._offer_bands(band, channels)
        if self.cmap not in self.param.cmap.objects:
            self.param.cmap.objects = [*CMAPS, self.cmap]
        self._offer_dates(raster)
        # Labels of each axis besides the grid and time, such as a level, and
        # the position shown along each.
        self.axis_labels = self._read_axes(raster)
        self.axis_positions = {axis: 0 for axis in self.axis_labels}
        self.fitted_clim = (0.0, 1.0)
        # The map's extent of this raster, in Web Mercator metres.
        self.bbox = to_web_mercator(raster.gs.geobox.extent)

        # 4. Drawing caches: pixels read at one zoom level around the view and
        #    the selection they show, the contrast applied, and the RGB overview
        self._pixel_windows = []
        self._cached_pixels = None
        self._cached_zoom = None
        self._cached_selection = None
        self._drawn_pixels = None
        self._drawn_image = None
        self._contrast_key = None
        self._contrasted_image = None
        self._overview = None
        self._overview_selection = None

    @staticmethod
    def _check_mappable(raster: xr.Dataset) -> None:
        """Refuse a raster a web map cannot place."""
        if raster.gs.geobox is None:
            raise ValueError("a map needs a raster with a CRS and spatial grid")
        bounds = raster.gs.bounds
        if raster.gs.crs.geographic and bounds.right > 180:
            # Web Mercator reads longitudes past 180° as the ones they wrap to,
            # so a 0-360° grid would draw and identify on the wrong ground.
            raise ValueError(
                f"longitudes run to {bounds.right:g}°; wrap them into -180..180 "
                'first: raster.assign_coords(x=(raster.x + 180) % 360 - 180).sortby("x")'
            )

    def _offer_bands(self, band: str | None, channels) -> None:
        """Offer each band, and the composite first when there is one; show `band`."""
        self.composite = None
        choices = list(self.bands)
        if channels:
            # The composite's label, unless a band already carries it
            self.composite = "RGB"
            while self.composite in self.bands:
                self.composite += " composite"
            choices = [self.composite, *choices]
        self.param.band.objects = choices
        self.rgb = tuple(channels) if channels else None
        self.band = band if band is not None else (self.composite or self.bands[0])
        self.param.band.allow_None = False

    def _offer_dates(self, raster: xr.Dataset) -> None:
        """Offer each date of the `time` axis, labelled to the second; show the first."""
        dates = list(raster.time.values) if "time" in raster.dims else []
        dates_by_label = {}
        for date in dates:
            dates_by_label[str(np.datetime_as_string(date, unit="s"))] = date
        self.param.time.objects = dates_by_label
        self.time = dates[0] if dates else None

    def _read_axes(self, raster: xr.Dataset) -> dict[str, list[str]]:
        """Return the labels of each band axis besides the grid and time."""
        axes = {}
        for name in self.bands:
            for axis, labels in raster[name].gs.axes.items():
                if axis == "time" or axis in axes:
                    continue
                if labels is None:
                    labels = np.arange(raster.sizes[axis])
                axes[axis] = [str(label) for label in labels]
        return axes

    @property
    def times(self) -> list[np.datetime64]:
        """Return the raster's dates, in order; empty when it has none."""
        if "time" not in self.data.dims:
            return []
        return list(self.data.time.values)

    def select_time(self, period: pd.Period) -> np.datetime64 | None:
        """Return the raster's last date inside `period`; None when it has none there."""
        times_in_period = []
        for time in self.times:
            if period.start_time <= pd.Timestamp(time) <= period.end_time:
                times_in_period.append(time)
        return times_in_period[-1] if times_in_period else None

    @property
    def summary(self) -> str:
        """Return the band count and pixel size, such as `3 bands · 10 m`."""
        grid = self.data.gs.geobox
        resolution = abs(grid.resolution.x)
        unit = "°" if grid.crs.geographic else f" {grid.crs.units[1]}"
        if unit == " metre":
            unit = " m"
        noun = "band" if len(self.bands) == 1 else "bands"
        return f"{len(self.bands)} {noun} · {resolution:.4g}{unit}"

    @property
    def caption(self) -> str | None:
        """Return the composite's channels, such as `RGB composite · B04 / B03 / B02`."""
        if not self.is_rgb:
            return None
        assert self.rgb is not None
        return "RGB composite · " + " / ".join(self.rgb)

    @property
    def swatches(self) -> dict[str, str] | None:
        """Return each class's colour by its name, when the band lists classes."""
        legend = self.legend
        if self.is_rgb or legend is None or not legend.classes:
            return None
        swatches = {}
        colors = legend.colors
        for code, title in legend.classes.items():
            swatches[str(title)] = colors[code]
        return swatches

    @property
    def ramp(self) -> tuple[str, str, tuple[float, float] | None] | None:
        """Return the band, colormap and limits of a continuous band."""
        if self.is_rgb or self.swatches is not None:
            return None
        return str(self.band), self.cmap, self.clim or self.fitted_clim

    def _create_copy(self):
        return RasterLayer(self.data, name=self.name, row=self.row, rgb=self.rgb)

    @property
    def is_rgb(self):
        """Return whether the declared or explicitly supplied composite is selected."""
        return self.band == self.composite

    @property
    def legend(self):
        """Return the selected band's declared class legend, if any."""
        return self.data[self.band].gs.legend if not self.is_rgb else None

    def select_shown_bands(self):
        """Select the current date, steps and bands without reading pixels."""
        if self.is_rgb:
            assert self.rgb is not None
            names = list(dict.fromkeys(self.rgb))
        else:
            names = [self.band]
        data = self.data[names]
        if "time" in data.dims:
            data = data.sel(time=self.time)
        steps = {}
        for axis, position in self.axis_positions.items():
            if axis in data.dims:
                steps[axis] = position
        return data.isel(steps)

    def read_view_pixels(self, view: GeoBox) -> xr.Dataset:
        """Read a padded viewport, reusing recent windows within `cache_bytes`.

        Windows share the current band/time/axis selection and use globally
        aligned Web Mercator zoom grids. Styling does not invalidate pixels.
        The active render may exceed the budget; oversized windows are not
        retained as historical windows. The active window still supports pans
        and styling. Rendered images and the RGB overview are separate.
        """
        selection = (
            self.band,
            self.rgb,
            self.time,
            tuple(sorted(self.axis_positions.items())),
        )
        if selection != self._cached_selection:
            self._pixel_windows.clear()
            self._cached_pixels = None
        self._cached_selection = selection
        zoom = find_zoom_level(view)
        bounds = view.boundingbox
        # The active image is already retained for rendering, even when it
        # exceeds the history budget. Styling and contained pans still reuse it.
        if self._cached_pixels is not None and self._cached_zoom == zoom:
            cached_bounds = self._cached_pixels.odc.geobox.boundingbox
            if (
                cached_bounds.left <= bounds.left
                and cached_bounds.bottom <= bounds.bottom
                and cached_bounds.right >= bounds.right
                and cached_bounds.top >= bounds.top
            ):
                return self._cached_pixels
        for position in range(len(self._pixel_windows) - 1, -1, -1):
            cached_zoom, cached = self._pixel_windows[position]
            cached_bounds = cached.odc.geobox.boundingbox
            if (
                cached_zoom == zoom
                and cached_bounds.left <= bounds.left
                and cached_bounds.bottom <= bounds.bottom
                and cached_bounds.right >= bounds.right
                and cached_bounds.top >= bounds.top
            ):
                self._pixel_windows.append(self._pixel_windows.pop(position))
                self._cached_pixels = cached
                self._cached_zoom = zoom
                return cached
        padded = bounds.buffered(bounds.span_x / 2, bounds.span_y / 2)
        pixels = self._read_pixels(create_zoom_grid(padded, zoom))
        self._cached_pixels = pixels
        self._cached_zoom = zoom
        if pixels.nbytes <= self.cache_bytes:
            self._pixel_windows.append((zoom, pixels))
            self._trim_cache()
        return pixels

    @param.depends("cache_bytes", watch=True)
    def _trim_cache(self):
        if not hasattr(self, "_pixel_windows"):
            return
        size = sum(pixels.nbytes for _, pixels in self._pixel_windows)
        while size > self.cache_bytes and self._pixel_windows:
            _, pixels = self._pixel_windows.pop(0)
            size -= pixels.nbytes

    def clear_cache(self) -> None:
        """Release viewport, rendered, contrast and overview buffers."""
        self._pixel_windows.clear()
        self._cached_pixels = None
        self._cached_selection = None
        self._cached_zoom = None
        self._drawn_pixels = None
        self._drawn_image = None
        self._contrast_key = None
        self._contrasted_image = None
        self._overview = None
        self._overview_selection = None

    def _read_pixels(self, geobox: GeoBox) -> xr.Dataset:
        """Read the shown bands onto `geobox`: from the row's files, or computed."""
        if self.row is None:
            return self.select_shown_bands().gs.read_pixels(geobox)
        names = list(self.select_shown_bands().data_vars)
        return self.row.gs.read_pixels(geobox, variables=names)

    def _draw_pixels(self, sampled):
        """Draw the sampled view: a band's values, its classes' colours, or a composite."""
        if not sampled.data_vars:
            return hv.Overlay([gv.Image([], crs=gv.tile_sources.OSM.crs)])
        x, y = sampled.x.values, sampled.y.values
        if self.is_rgb:
            # The channels as read; `_stretch` maps them onto colours
            assert self.rgb is not None
            red, green, blue = (sampled[name].values for name in self.rgb)
            shown = np.isfinite(np.stack([red, green, blue])).all(axis=0)
            return hv.Overlay(
                [_create_rgb(x, y, red, green, blue, shown.astype("float32"))]
            )
        legend = self.legend
        if legend is not None and legend.classes:
            # Classes take their legend's colours, as `.gs.plot` draws them;
            # a pixel no class names is transparent.
            classes = stamp_attrs(sampled[self.band], legend)
            red, green, blue = colorize(classes).values
            shown = np.isfinite(red).astype("float32")
            red, green, blue = (
                np.nan_to_num(channel) for channel in (red, green, blue)
            )
            return hv.Overlay([_create_rgb(x, y, red, green, blue, shown)])
        # A continuous band's values, which the colormap colours
        values = sampled[self.band].values
        return hv.Overlay(
            [gv.Image((x, y, values), crs=gv.tile_sources.OSM.crs, vdims=[self.band])]
        )

    def draw(self, view: GeoBox | None) -> hv.Overlay:
        """Return the view's pixels in the layer's contrast and style.

        Pixels are read again only past the cache, and stretched again only
        when they or the contrast change. None draws the whole raster.
        """
        if view is None:
            box = self.bbox
            view = create_view(
                (box.left, box.right), (box.bottom, box.top), DEFAULT_VIEW_SHAPE
            )
        pixels = self.read_view_pixels(view)
        # The same pixels draw the same element, which the map does not resend.
        if pixels is not self._drawn_pixels:
            self._drawn_image = self._draw_pixels(pixels)
            self._drawn_pixels = pixels
        contrasted = self._stretch(
            self._drawn_image, self.clim, self.stretch_percentiles
        )
        return self._style(contrasted)

    def _get_rgb_limits(self, stretch):
        """Fit each channel to a cached scene overview, independent of the view."""
        selection = (self.rgb, self.time, tuple(self.axis_positions.items()))
        if self._overview is None or self._overview_selection != selection:
            grid = self.data.gs.geobox
            target = GeoBox.from_bbox(
                grid.boundingbox,
                shape=(min(OVERVIEW_SIZE, grid.height), min(OVERVIEW_SIZE, grid.width)),
                tight=True,
            )
            self._overview = self._read_pixels(target)
            self._overview_selection = selection
        assert self.rgb is not None
        limits = []
        for name in self.rgb:
            limits.append(_fit_limits(self._overview[name].values, stretch))
        return limits

    def _stretch(self, plot, clim, percentiles):
        """Apply the contrast: per channel for a composite, a colormap range for a band.

        The last result is reused while the pixels and contrast are unchanged.
        """
        element = plot.values()[0]
        data = element.data
        if self._contrast_key is not None and self._contrasted_image is not None:
            previous, limits, previous_percentiles = self._contrast_key
            if (
                previous is data
                and limits == clim
                and previous_percentiles == percentiles
            ):
                return self._contrasted_image
        if self.is_rgb:
            plot = self._stretch_composite(element, clim, percentiles)
        elif not (self.legend and self.legend.classes):
            # A band's range comes from the view's pixels; classes need none.
            values = element.dimension_values(element.vdims[0], flat=False)
            self.fitted_clim = _fit_limits(values, percentiles)
        self._contrasted_image = cast(
            "hv.Overlay",
            plot.opts(hv.opts.Image(clim=clim or self.fitted_clim, backend="bokeh")),
        )
        self._contrast_key = (data, clim, percentiles)
        return self._contrasted_image

    def _stretch_composite(self, element, clim, stretch):
        """Stretch each channel of a composite onto [0, 1], by `clim` or the overview."""
        channels = np.stack(
            [element.dimension_values(dim, flat=False) for dim in element.vdims[:3]]
        )
        if not channels.size:
            return hv.Overlay([element])
        limits = [clim] * 3 if clim is not None else self._get_rgb_limits(stretch)
        self.fitted_clim = (
            min(low for low, _ in limits),
            max(high for _, high in limits),
        )
        pixels = np.empty_like(channels, dtype="float32")
        for position, channel_limits in enumerate(limits):
            pixels[position] = stretch_channels(channels[position], channel_limits)
        x = element.dimension_values(element.kdims[0], expanded=False)
        y = element.dimension_values(element.kdims[1], expanded=False)
        shown = element.dimension_values("A", flat=False)
        return hv.Overlay([element.clone((x, y, *pixels, shown))])

    def _style(self, plot: hv.Overlay) -> hv.Overlay:
        styled = plot.opts(
            hv.opts.Image(
                alpha=self.alpha,
                visible=self.visible,
                cmap=self.cmap,
                # Name the value only: the map's x and y are Web Mercator metres.
                tools=[HoverTool(tooltips=[(self.band, "@image")])],
                backend="bokeh",
            ),
            hv.opts.RGB(alpha=self.alpha, visible=self.visible, backend="bokeh"),
        )
        return cast("hv.Overlay", styled)

    def identify(self, x, y, tolerance=0):
        """Read one source pixel at a map click."""
        from pyproj import Transformer

        grid = self.data.gs.geobox
        x, y = Transformer.from_crs(3857, grid.crs, always_xy=True).transform(x, y)
        bounds = grid.boundingbox
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            return {}
        data = (
            self.select_shown_bands()
            .gs.to_nan()
            .sel(x=x, y=y, method="nearest")
            .compute()
        )
        return {str(name): str(value.item()) for name, value in data.data_vars.items()}

    def get_info(self):
        """Return the source grid and band metadata without computing pixels."""
        grid = self.data.gs.geobox
        return {
            "CRS": self.data.gs.crs_name,
            "Size": f"{grid.width} × {grid.height}",
            "Bands": self.bands,
            "RGB": self.rgb,
            "Attributes": dict(self.data.attrs),
            "Variables": {name: dict(self.data[name].attrs) for name in self.bands},
        }
