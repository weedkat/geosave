"""Raster extension: a band's stored type, fill, packing and unit."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TypedDict, cast

import pystac
import xarray as xr
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

import geosave.geodata.attrs as attrs
from geosave.geodata.stac.stamps import LoadSource, register_stamp


def read_json_number(value: float) -> float:
    """Spell a float as the shortest number that reads back as its own dtype.

    A float32 scale of `1e-4` is `9.999999747378752e-05` as a Python float;
    its own shortest spelling, `1e-04`, is what a reader expects.
    """
    return float(str(value))


def build_band(variable: xr.DataArray) -> RasterBand:
    """Build the Raster band storing one variable.

    Args:
        variable: Variable as `open_raster` opens it, holding stored values.
            No pixel is read.

    Returns:
        Its band: `data_type`, `nodata`, `scale` and `offset`, and `unit`.

    Examples:
        >>> build_band(scene.red).scale
        0.0001
    """
    fill = variable.gs.nodata
    if isinstance(fill, float) and not isfinite(fill):
        fill = NoDataStrings(str(fill))
    dtype = variable.dtype.name

    # Packing as stored, in its own dtype, spelled as short as it reads back.
    scale = variable.attrs.get("scale_factor")
    offset = variable.attrs.get("add_offset")
    return RasterBand.create(
        nodata=fill,
        data_type=DataType(dtype) if dtype in list(DataType) else DataType.OTHER,
        scale=None if scale is None else read_json_number(scale),
        offset=None if offset is None else read_json_number(offset),
        unit=variable.attrs.get("units"),
    )


class RasterAttrs(TypedDict, total=False):
    """CF storage attrs translated from one Raster band."""

    scale_factor: float
    add_offset: float
    units: str


def read_attrs(band: RasterBand) -> RasterAttrs:
    """Translate a band's packing and units into CF attrs.

    Stored dtype and nodata belong to the file reader and are not copied.
    """
    fields: RasterAttrs = {}
    if band.scale is not None or band.offset is not None:
        packing = attrs.CFPacking(
            scale_factor=1.0 if band.scale is None else band.scale,
            add_offset=0.0 if band.offset is None else band.offset,
        )
        fields = cast("RasterAttrs", packing.to_attrs())
    if band.unit is not None:
        fields["units"] = band.unit
    return fields


def write_asset(asset: pystac.Asset, bands: list[RasterBand]) -> None:
    """Write storage bands and declare their schema on the owner."""
    RasterExtension.ext(asset, add_if_missing=True).bands = bands


def read_item_attrs(item: pystac.Item, name: str) -> RasterAttrs:
    """Translate the Raster fields of the band ODC loads as `name` into CF attrs."""
    from geosave.geodata.stac.band import read_band  # band imports this module

    fields = {}
    for key, value in read_band(item, name).items():
        fields[key.removeprefix("raster:")] = value
    return read_attrs(RasterBand(fields))


def check_packing(name: str, items: Sequence[pystac.Item]) -> None:
    """Refuse Items that pack one band with different scale or offset.

    Raises:
        ValueError: If two Items state different `(scale, offset)` for `name`.
    """
    expected = None
    for item in items:
        packing = attrs.CFPacking.model_validate(read_item_attrs(item, name))

        # Omitted scale/offset mean identity. Nodata is excluded: ODC remaps
        # source sentinels on loading.
        scale = (packing.scale_factor, packing.add_offset)
        if expected is not None and scale != expected:
            raise ValueError(
                f"band {name!r} has incompatible source scale/offset: "
                f"{expected} and {scale} in Item {item.id!r}; "
                "load and decode the sources separately before combining them"
            )
        expected = scale


@register_stamp("raster")
def stamp_raster(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each loaded band's packing, units and nodata declaration.

    The variable's own attrs, as ODC set them, win over STAC's. Units are kept
    only when every Item agrees.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with `scale_factor`, `add_offset` and `units` stated, and
        nodata written to match the fill ODC cast into the output dtype.

    Raises:
        ValueError: If Items pack a loaded band with different scale/offset.
    """
    stamped = {}
    for name, variable in loaded.data_vars.items():
        check_packing(str(name), source.items)

        observations = []
        for item in source.items:
            observations.append(xr.Dataset(attrs=read_item_attrs(item, str(name))))
        stated = xr.merge(observations, combine_attrs="drop_conflicts").attrs

        variable = variable.assign_attrs({**stated, **variable.attrs})
        fill = variable.attrs.pop("nodata", variable.attrs.get("_FillValue"))
        if fill is not None:
            variable = variable.rio.write_nodata(fill)
        stamped[name] = variable
    return loaded.assign(stamped)
