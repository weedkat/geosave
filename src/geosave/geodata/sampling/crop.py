"""Sample spatial windows from georeferenced xarray objects."""

import random
from operator import index

import xarray as xr

from geosave.geodata.errors import RasterValidationError


def random_crop[T: (xr.DataArray, xr.Dataset, xr.DataTree)](
    obj: T, size: int | tuple[int, int], seed: int | None = None
) -> T:
    """Draw an in-bounds pixel window uniformly and return the cropped raster.

    Args:
        obj: A raster or a DataTree whose raster nodes share one grid.
        size: Side length in pixels, or a (height, width) pair.
        seed: An integer to reproduce one draw. When omitted, draws advance
            Python's shared random generator.

    Returns:
        The same kind of xarray object on the selected grid, retaining its
        acquisition coordinates and lazy pixel arrays.

    Raises:
        RasterValidationError: The input lacks a suitable grid.
        ValueError: The requested window is empty or does not fit.

    Examples:
        >>> chip = gs.random_crop(image, size=128, seed=42)
        >>> target = label.gs.crop_to_grid(chip.gs.geobox)
    """
    grid = obj.gs.common_geobox() if isinstance(obj, xr.DataTree) else obj.gs.geobox
    if grid is None:
        raise RasterValidationError("A random crop requires a raster grid.")

    rows, cols = size if isinstance(size, tuple) else (size, size)
    height, width = index(rows), index(cols)
    if not (0 < height <= grid.height and 0 < width <= grid.width):
        raise ValueError(
            f"Crop {(height, width)} must fit inside grid {tuple(grid.shape)}."
        )

    rng = random if seed is None else random.Random(seed)
    top = rng.randint(0, grid.height - height)
    left = rng.randint(0, grid.width - width)
    return obj.gs.crop_to_grid(grid[top : top + height, left : left + width])
