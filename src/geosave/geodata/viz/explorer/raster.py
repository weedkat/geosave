"""Raster display choices and viewport reads for the interactive map."""

from __future__ import annotations

import math
from collections import defaultdict

from dask.base import is_dask_collection, tokenize

import geoviews as gv
import holoviews as hv
import numpy as np
import panel as pn
import param
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from odc.geo.xr import xr_coords
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject
from rasterio.windows import from_bounds, transform

from geosave.geodata.io import open_raster
from geosave.geodata.viz.style import (
    CATEGORY_PALETTE,
    CMAPS,
    DEFAULT_CMAP,
    Stretch,
    to_rgb255,
)


def read_window(data: xr.Dataset, target: GeoBox) -> xr.Dataset:
    """Read selected variables in a viewport, sampling before computing pixels.

    Unmodified file variables use Rasterio windows and overviews. Computed
    arrays use spatial indexing before Dask execution. A computed overview can
    still touch every source chunk; this does not build a pyramid for it.
    """
    grid = data.odc.geobox
    bounds = target.extent.to_crs(grid.crs).boundingbox
    extent = grid.boundingbox
    left, bottom = max(bounds.left, extent.left), max(bounds.bottom, extent.bottom)
    right, top = min(bounds.right, extent.right), min(bounds.top, extent.top)
    if left >= right or bottom >= top:
        return xr.Dataset(coords=xr_coords(target))
    # Bound source sampling by the viewport's resolution before reprojection.
    width, height = min(1200, target.width), min(900, target.height)
    files = defaultdict(list)
    computed = []
    for name, variable in data.data_vars.items():
        source = variable.encoding.get("source")
        if (
            source
            and "band" in variable.encoding
            and grid.axis_aligned
            and is_dask_collection(variable)
        ):
            decoded = any(
                key in variable.encoding
                for key in ("_FillValue", "scale_factor", "add_offset")
            )
            files[str(source), decoded].append(str(name))
        else:
            computed.append(str(name))
    values = {}
    for (source, decoded), names in files.items():
        # Encoding records origin, even after a caller replaces the pixels.
        # Dask's public token identifies the unchanged reader expression.
        try:
            original = open_raster(source, mask_and_scale=decoded)
        except RasterioIOError:
            # Persisted Dask pixels can outlive the file recorded in encoding.
            computed.extend(names)
            continue
        stored = {
            int(value.encoding["band"]): value for value in original.data_vars.values()
        }
        unchanged = []
        try:
            for name in names:
                variable = data[name]
                band = int(variable.encoding["band"])
                if band in stored and tokenize(variable.data) == tokenize(
                    stored[band].data
                ):
                    unchanged.append(name)
                else:
                    computed.append(name)
        finally:
            original.close()
        names = unchanged
        if not names:
            continue
        with rasterio.open(source) as file:
            # A crop, reprojection or replaced grid must use its xarray values.
            if (
                file.crs != grid.crs
                or file.transform != grid.affine
                or (file.height, file.width) != tuple(grid.shape)
            ):
                computed.extend(names)
                continue
            window = from_bounds(left, bottom, right, top, file.transform)
            shape = (
                min(height, max(1, math.ceil(window.height))),
                min(width, max(1, math.ceil(window.width))),
            )
            indexes = [int(data[name].encoding["band"]) for name in names]
            # Current xarray metadata may replace or clear the file's sentinel.
            pixels = file.read(
                indexes,
                window=window,
                out_shape=(len(names), *shape),
                masked=decoded,
                resampling=Resampling.nearest,
            ).astype("float32")
            if decoded:
                pixels = pixels.filled(np.nan)
            affine = transform(window, file.transform) * file.transform.scale(
                window.width / shape[1], window.height / shape[0]
            )
            sample_grid = GeoBox(shape, affine, grid.crs)
            for index, name in enumerate(names):
                variable = data[name]
                # Decoding is part of the reader expression. Rioxarray can
                # give raw and decoded arrays the same Dask token.
                encoding = stored[int(variable.encoding["band"])].encoding
                if decoded:
                    pixels[index] = pixels[index] * encoding.get(
                        "scale_factor", 1
                    ) + encoding.get("add_offset", 0)
                nodata = variable.gs.nodata
                if nodata is not None:
                    pixels[index][pixels[index] == nodata] = np.nan
                values[name] = xr.DataArray(
                    pixels[index], dims=("y", "x"), coords=xr_coords(sample_grid)
                )
    if computed:
        geometry = BoundingBox(left, bottom, right, top, crs=grid.crs).polygon
        window = data[computed].odc.crop(geometry)
        steps = {
            "y": slice(None, None, max(1, math.ceil(window.sizes["y"] / height))),
            "x": slice(None, None, max(1, math.ceil(window.sizes["x"] / width))),
        }
        sample = window.isel(steps).gs.to_nan().astype("float32").compute()
        for name in computed:
            values[name] = sample[name]
    projected = {}
    for name, variable in values.items():
        source_grid = variable.odc.geobox
        pixels = np.full(tuple(target.shape), np.nan, dtype="float32")
        reproject(
            variable.transpose("y", "x").values,
            pixels,
            src_transform=source_grid.affine,
            src_crs=str(source_grid.crs),
            src_nodata=np.nan,
            dst_transform=target.affine,
            dst_crs=str(target.crs),
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        projected[name] = xr.DataArray(
            pixels, dims=("y", "x"), coords=xr_coords(target)
        )
    return xr.Dataset(projected, coords=xr_coords(target))


class RasterLayer(param.Parameterized):
    """Keep one raster's display choices without changing its pixels or metadata."""

    visible = param.Boolean(True)
    alpha = param.Number(0.8, bounds=(0.0, 1.0), label="Opacity")
    display = param.Selector(default=None, allow_None=True, label="Display")
    rgb = param.Tuple(default=None, length=3, allow_None=True, precedence=-1)
    cmap = param.Selector(
        default=DEFAULT_CMAP, objects=CMAPS, check_on_set=False, label="Colormap"
    )
    stretch = param.Range((2.0, 98.0), bounds=(0.0, 100.0), label="Stretch percentiles")
    clim = param.Range(None, allow_None=True, label="Color limits")
    limits = param.Range((0.0, 1.0), precedence=-1)
    time = param.Selector(default=None, allow_None=True, label="Date")

    def __init__(self, raster, band=None, *, rgb=None, name=None, cmap=None, alpha=0.8):
        if isinstance(raster, xr.DataArray):
            raster = (
                raster.gs.to_raster()
                if "band" in raster.dims
                else raster.to_dataset(name=raster.name or "band")
            )
        if raster.odc.geobox is None:
            raise ValueError("a map needs a raster with a CRS and spatial grid")
        if band is not None and rgb is not None:
            raise ValueError("choose a band or RGB, not both")
        self.data = raster
        self.names = [
            str(name)
            for name, value in raster.data_vars.items()
            if {"y", "x"} <= set(value.dims)
        ]
        if not self.names:
            raise ValueError("the raster has no spatial bands")
        for choice in [band, *(rgb or ())]:
            if choice is not None and choice not in self.names:
                raise KeyError(f"{choice!r} is not a spatial band")
        channels = rgb or raster.gs.rgb
        if channels is not None and any(name not in self.names for name in channels):
            raise KeyError("the declared RGB includes a non-spatial band")
        super().__init__(
            name=name or str(band or ("RGB" if channels else self.names[0])),
            alpha=alpha,
            cmap=cmap or DEFAULT_CMAP,
        )
        self.composite = None
        choices = list(self.names)
        if channels:
            self.composite = "RGB"
            while self.composite in self.names:
                self.composite += " composite"
            choices = [self.composite, *choices]
        self.param.display.objects = choices
        self.rgb = tuple(channels) if channels else None
        if self.cmap not in self.param.cmap.objects:
            self.param.cmap.objects = [*CMAPS, self.cmap]
        self.display = band if band is not None else (self.composite or self.names[0])
        self.param.display.allow_None = False
        dates = list(raster.time.values) if "time" in raster.dims else []
        self.param.time.objects = {
            str(np.datetime_as_string(date, unit="s")): date for date in dates
        }
        self.time = dates[0] if dates else None
        self.limits = (0.0, 1.0)
        self._plot = None

    @property
    def bbox(self):
        """Return the raster extent in Web Mercator without reading pixels."""
        return self.data.odc.geobox.extent.to_crs("EPSG:3857").boundingbox

    @property
    def is_rgb(self):
        """Return whether the declared or explicitly supplied composite is selected."""
        return self.display == self.composite

    @property
    def legend(self):
        """Return the selected band's declared class legend, if any."""
        return self.data[self.display].gs.legend if not self.is_rgb else None

    @property
    def colors(self) -> dict[int, str]:
        """Use declared class colors, assigning a palette to uncolored classes."""
        legend = self.legend
        if legend is None or not legend.classes:
            return {}
        palette = CATEGORY_PALETTE["light"]
        colors = {}
        for index, code in enumerate(legend.classes):
            colors[code] = (legend.color_map or {}).get(
                code, palette[index % len(palette)]
            )
        return colors

    def select(self):
        """Select the current date and bands without reading pixels."""
        if self.is_rgb:
            assert self.rgb is not None
            names = list(dict.fromkeys(self.rgb))
        else:
            names = [self.display]
        data = self.data[names]
        if "time" in data.dims:
            data = data.sel(time=self.time)
        return data

    def render(self, x_range, y_range, **_):
        """Sample visible pixels as native elements, before applying contrast."""
        box = self.bbox
        x_range = x_range or (box.left, box.right)
        y_range = y_range or (box.bottom, box.top)
        target = GeoBox.from_bbox(
            (x_range[0], y_range[0], x_range[1], y_range[1]),
            crs="EPSG:3857",
            shape=(600, 900),
            tight=True,
        )
        sampled = read_window(self.select(), target)
        if not sampled.data_vars:
            return hv.Overlay([gv.Image([], crs=gv.tile_sources.OSM.crs)])
        x, y = sampled.x.values, sampled.y.values
        if not self.is_rgb:
            values = sampled[self.display].values
            legend = self.legend
            if legend is not None and legend.classes:
                rgba = np.zeros((*values.shape, 4), dtype="float32")
                for code, color in self.colors.items():
                    rgba[values == code] = (
                        *np.asarray(to_rgb255(str(color))) / 255,
                        1.0,
                    )
                return hv.Overlay(
                    [
                        gv.RGB(
                            (
                                x,
                                y,
                                rgba[..., 0],
                                rgba[..., 1],
                                rgba[..., 2],
                                rgba[..., 3],
                            ),
                            crs=gv.tile_sources.OSM.crs,
                            vdims=["R", "G", "B", "A"],
                        )
                    ]
                )
            return hv.Overlay(
                [
                    gv.Image(
                        (x, y, values),
                        crs=gv.tile_sources.OSM.crs,
                        vdims=[self.display],
                    )
                ]
            )
        assert self.rgb is not None
        channels = np.stack([sampled[name].values for name in self.rgb])
        rgba = np.concatenate(
            [channels, np.expand_dims(np.isfinite(channels).all(axis=0), axis=0)],
            axis=0,
        )
        return hv.Overlay(
            [
                gv.RGB(
                    (x, y, rgba[0], rgba[1], rgba[2], rgba[3]),
                    crs=gv.tile_sources.OSM.crs,
                    vdims=["R", "G", "B", "A"],
                )
            ]
        )

    def draw(self, ranges):
        """Bind pixel reads separately from glyph-only style changes."""
        if self._plot is None:
            draw = pn.bind(
                self.render,
                display=self.param.display,
                time=self.param.time,
            )
            pixels = hv.DynamicMap(draw, streams=[ranges])
            self._plot = pixels.apply(
                self._style,
                alpha=self.param.alpha,
                visible=self.param.visible,
                cmap=self.param.cmap,
                clim=self.param.clim,
                stretch=self.param.stretch,
            )
        return self._plot

    def _style(self, plot, alpha, visible, cmap, clim, stretch):
        element = plot.values()[0]
        if self.is_rgb or not (self.legend and self.legend.classes):
            dimensions = element.vdims[:3] if self.is_rgb else element.vdims[:1]
            channels = np.stack(
                [element.dimension_values(dim, flat=False) for dim in dimensions]
            )
            finite = channels[np.isfinite(channels)]
            low, high = np.percentile(finite, stretch) if finite.size else (0.0, 1.0)
            self.limits = (float(low), float(high) if high > low else float(low) + 1)
            if self.is_rgb and channels.size:
                pixels = Stretch(clim or self.limits).apply(channels)
                element = element.clone(
                    (
                        element.dimension_values(element.kdims[0], expanded=False),
                        element.dimension_values(element.kdims[1], expanded=False),
                        *pixels,
                        element.dimension_values("A", flat=False),
                    )
                )
                plot = hv.Overlay([element])
        return plot.opts(
            hv.opts.Image(
                alpha=alpha,
                visible=visible,
                cmap=cmap,
                clim=clim or self.limits,
                tools=["hover"],
                backend="bokeh",
            ),
            hv.opts.RGB(alpha=alpha, visible=visible, backend="bokeh"),
        )

    def identify(self, x, y, tolerance=0):
        """Read one source pixel at a map click."""
        from pyproj import Transformer

        grid = self.data.odc.geobox
        x, y = Transformer.from_crs(3857, grid.crs, always_xy=True).transform(x, y)
        bounds = grid.boundingbox
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            return {}
        data = self.select().gs.to_nan().sel(x=x, y=y, method="nearest").compute()
        return {str(name): str(value.item()) for name, value in data.data_vars.items()}

    def get_info(self):
        """Return the source grid and band metadata without computing pixels."""
        grid = self.data.odc.geobox
        return {
            "CRS": f"EPSG:{grid.crs.epsg}" if grid.crs.epsg else str(grid.crs),
            "Size": f"{grid.width} × {grid.height}",
            "Bands": self.names,
            "RGB": self.rgb,
            "Attributes": dict(self.data.attrs),
            "Variables": {name: dict(self.data[name].attrs) for name in self.names},
        }
