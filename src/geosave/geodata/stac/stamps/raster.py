"""Stamp each loaded band with its declared STAC Raster packing and units."""

from collections.abc import Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig
from pystac.extensions.raster import RasterBand

from geosave.geodata.stac.extensions import raster

from .bands import find_band_sources, read_band_fields


def stamp_nodata(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Write each raster band's effective ODC output nodata as its CF fill value."""
    bands = {}

    for name, band in ds.data_vars.items():
        fill = band.attrs.get("nodata")
        if fill is None or band.gs.spatial_dims is None:
            continue

        bands[name] = band.gs.write_nodata(fill)

    return ds.assign(bands)


def stamp_raster(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Write the scale, offset and unit declared for each loaded band.

    A variable has one scale and offset for all its pixels, including pixels
    fused from several Items. Its fill comes from the loaded variable.

    Raises:
        ValueError: Items supplying the same band use different scale or offset.
    """
    stamped = {}
    for variable, sources in find_band_sources(ds, items, stac_cfg):
        statements = []
        expected = None
        for item, asset, index in sources:
            fields = {}
            for key, value in read_band_fields(asset, index).items():
                fields[key.removeprefix("raster:")] = value

            statement = raster.read_attrs(RasterBand(fields))
            # An existing band with no scale or offset stores unscaled values.
            scale = (
                statement.get("scale_factor", 1.0),
                statement.get("add_offset", 0.0),
            )
            if expected is not None and scale != expected:
                raise ValueError(
                    f"band {variable.name!r} has incompatible source scale/offset: "
                    f"{expected} and {scale} in Item {item.id!r}; "
                    "load and decode the sources separately before combining them"
                )

            expected = scale
            statements.append(xr.Dataset(attrs=statement))

        agreed = xr.merge(statements, combine_attrs="drop_conflicts").attrs
        stamped[variable.name] = variable.assign_attrs({**agreed, **variable.attrs})

    return ds.assign(stamped)
