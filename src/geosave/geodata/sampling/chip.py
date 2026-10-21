"""Cut sample footprints into windows on a chosen pixel grid with Tiler."""

from __future__ import annotations


# Work behind the tables' operations; call those instead.
__all__: list[str] = []

from collections.abc import Mapping
from typing import TYPE_CHECKING

import geopandas as gpd

from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.math import snap_grid
from tiler import Tiler


from geosave.geodata.core.vector.samples import GeoSamples

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame
    from geosave.geodata.transform.extent import Padding
    from geosave.geodata.transform.warp import Resampling


def create_tiler(
    region_shape: tuple[int, int],
    chip_shape: tuple[int, int],
    *,
    overlap: int | float,
    pad_mode: Padding,
) -> tuple[Tiler, list[tuple[int, int]]]:
    """Lay chips over a region, with a halo around it when chips overlap.

    The halo keeps every region pixel away from a chip edge at least once, so
    a tapered merge leaves no seam at the region's border.

    Returns:
        The Tiler, and the halo pixels before and after `y`, then `x`.

    Examples:
        >>> tiler, halo_widths = create_tiler((300, 500), (128, 128), overlap=32, pad_mode="reflect")
        >>> len(tiler), halo_widths
        (24, [(48, 48), (48, 48)])
    """
    tiler = Tiler(region_shape, chip_shape, overlap=overlap, mode=pad_mode)
    halo_widths = [(0, 0), (0, 0)]
    if overlap:
        padded_shape, halo_widths = tiler.calculate_padding()
        tiler.recalculate(data_shape=padded_shape)
    return tiler, [(int(before), int(after)) for before, after in halo_widths]


def place_chip(
    region: GeoBox,
    tiler: Tiler,
    halo_widths: list[tuple[int, int]],
    chip_number: int,
) -> tuple[GeoBox, list[list[int]]]:
    """Place one Tiler chip on the ground.

    Returns:
        The chip's grid, and its pixels past the region before and after
        `y`, then `x`.
    """
    first_corner, last_corner = tiler.get_tile_bbox(chip_number)
    top = int(first_corner[0]) - halo_widths[0][0]
    left = int(first_corner[1]) - halo_widths[1][0]
    height = int(last_corner[0] - first_corner[0])
    width = int(last_corner[1] - first_corner[1])
    chip_grid = GeoBox(
        (height, width),
        region.transform * Affine.translation(left, top),
        region.crs,
    )
    region_height, region_width = region.shape
    pad_widths = [
        [max(0, -top), max(0, top + height - region_height)],
        [max(0, -left), max(0, left + width - region_width)],
    ]
    return chip_grid, pad_widths


def chip(
    samples: gpd.GeoDataFrame,
    size: int | tuple[int, int],
    *,
    target: GeoBox | None,
    resolution: float | None,
    resampling: Resampling | Mapping[str, Resampling],
    overlap: int | float,
    pad_mode: Padding,
) -> GeoDataFrame[GeoSamples]:
    """Implement `GeoSamples.chip` without changing its public table contract."""
    if target is not None and not isinstance(target, GeoBox):
        raise TypeError("target must be a GeoBox; use resolution to plan local grids")
    if resolution is not None and target is not None:
        raise ValueError("target already fixes resolution; supply only one")
    chip_shape = (size, size) if isinstance(size, int) else size
    layouts = {}
    records = []
    for _, row in samples.iterrows():
        grid = target if target is not None else row.gs.geobox
        if resolution is not None:
            grid = row.gs._create_grid(resolution)
        if grid is None:
            raise ValueError(
                "chip requires resolution or target=grid for an unchipped sample"
            )
        # WGS84 footprints can return a fraction of a pixel past their original
        # edges. ODC's snapping tolerance avoids manufacturing an extra border.
        bounds = grid.project(row.gs.region).boundingbox
        left, width = snap_grid(bounds.left, bounds.right, 1, tol=1e-7)
        top, height = snap_grid(bounds.bottom, bounds.top, 1, tol=1e-7)
        region = GeoBox(
            (height, width), grid.transform * Affine.translation(left, top), grid.crs
        )
        region_shape = (int(region.shape[0]), int(region.shape[1]))
        if region_shape not in layouts:
            layouts[region_shape] = create_tiler(
                region_shape, chip_shape, overlap=overlap, pad_mode=pad_mode
            )
        tiler, halo = layouts[region_shape]

        for number in range(len(tiler)):
            grid, padding = place_chip(region, tiler, halo, number)
            records.append(
                row.gs._to_chip(
                    grid,
                    region=region,
                    pad_widths=padding,
                    pad_mode=pad_mode,
                    resampling=resampling,
                    geometry=samples.geometry.name,
                )
            )

    return GeoSamples.from_rows(
        records, geometry=samples.geometry.name, attrs=samples.attrs
    )
