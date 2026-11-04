"""Cut sample trees into model-sized chips, and merge per-chip outputs back."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from tiler import Merger, Tiler



class Chips:
    """Cut trees into fixed-size chips, one grid each, and merge per-chip outputs back.

    Each tree is laid out on its ground at `resolution`, in its groups' CRS,
    and cut into `size`-pixel chips. Overlapping chips get a halo around the
    tree, so a tapered merge weights every pixel of the tree away from a chip
    edge at least once. The same layout cuts and merges.

    Args:
        trees: One tree per sample, such as `to_datatree` returns; a label may be
            one more group.
        size: Chip width and height in pixels.
        resolution: Pixel size of the chips, in the units of the trees' CRS.
        overlap: Pixels neighbouring chips share.

    Raises:
        ValueError: A tree's groups lie in different CRSs, or a size is not
            positive.

    Examples:
        >>> chips = Chips(trees, size=256, resolution=10, overlap=32)
        >>> tree_index, chip = chips[7]
        >>> chips.add(7, model(chip.gs.to_tensor()))
        >>> rasters = chips.merge()
    """

    def __init__(
        self,
        trees: Sequence[xr.DataTree],
        size: int,
        resolution: float,
        overlap: int = 0,
    ) -> None:
        """Lay each tree's ground into chips, remembering which tree each comes from."""
        if size < 1 or resolution <= 0 or not 0 <= overlap < size:
            raise ValueError(
                f"chips need size >= 1, resolution > 0 and 0 <= overlap < size; "
                f"got size={size}, resolution={resolution}, overlap={overlap}"
            )
        self.trees = list(trees)
        self.size = size
        self.overlap = overlap
        self.grids: list[GeoBox] = []
        self.tilers: list[Tiler] = []
        self.halos: list[list[tuple[int, int]]] = []
        # Chip index -> (tree index, tile id in that tree's layout).
        self.chip_sources: list[tuple[int, int]] = []
        self.mergers: dict[int, Merger] = {}

        for tree_index, tree in enumerate(self.trees):
            grid = lay_grid(tree, resolution)
            height, width = grid.shape
            tiler, halo = lay_tiles((height, width), size, overlap)
            self.grids.append(grid)
            self.tilers.append(tiler)
            self.halos.append(halo)
            for tile_id in range(len(tiler)):
                self.chip_sources.append((tree_index, tile_id))

    def __len__(self) -> int:
        """Return the number of chips over all trees."""
        return len(self.chip_sources)

    def __getitem__(self, index: int) -> tuple[int, xr.DataTree]:
        """Return the index of the chip's source tree, and the chip.

        Args:
            index: Chip index.

        Returns:
            The source tree's position in `trees`, and that tree aligned lazily
            onto the chip's grid.
        """
        tree_index, _ = self.chip_sources[index]
        chip = self.trees[tree_index].gs.align(target=self.grid(index))
        return tree_index, cast("xr.DataTree", chip)

    def grid(self, index: int) -> GeoBox:
        """Return a chip's grid: its CRS, transform and shape.

        Args:
            index: Chip index.

        Returns:
            The `size` × `size` grid, reaching past the tree's edge where the
            chip pads.
        """
        tree_index, tile_id = self.chip_sources[index]
        tree_grid = self.grids[tree_index]
        halo = self.halos[tree_index]
        first_corner, _ = self.tilers[tree_index].get_tile_bbox(tile_id)
        # The layout counts pixels from the halo's corner; the grid from the tree's.
        top = int(first_corner[0]) - halo[0][0]
        left = int(first_corner[1]) - halo[1][0]
        transform = tree_grid.transform * Affine.translation(left, top)
        return GeoBox((self.size, self.size), transform, tree_grid.crs)

    def add(self, index: int, output: np.ndarray) -> None:
        """Accumulate the output of chip `index` into its source tree's raster.

        Args:
            index: Chip index.
            output: ``(size, size)`` or ``(channels, size, size)`` values on the
                chip's grid.
        """
        tree_index, tile_id = self.chip_sources[index]
        channel_output = output if output.ndim == 3 else output[np.newaxis]
        if tree_index not in self.mergers:
            self.mergers[tree_index] = self.create_merger(
                tree_index, channel_output.shape[0]
            )
        self.mergers[tree_index].add(tile_id, channel_output)

    def create_merger(self, tree_index: int, channel_count: int) -> Merger:
        """Return a Merger on a tree's layout, for outputs of `channel_count` channels.

        Args:
            tree_index: Tree the outputs belong to.
            channel_count: Channels each output holds.

        Returns:
            A Merger tapering overlaps with a Hann window, or averaging without overlap.
        """
        height, width = self.grids[tree_index].shape
        tiler = Tiler(
            data_shape=(channel_count, height, width),
            tile_shape=(channel_count, self.size, self.size),
            overlap=self.overlap,
            channel_dimension=0,
        )
        if self.overlap:
            padded_shape, _ = tiler.calculate_padding()
            tiler.recalculate(data_shape=padded_shape)
        window = "hann" if self.overlap else None
        return Merger(tiler, window=window)

    def merge(self) -> list[xr.DataArray]:
        """Return one merged raster per tree, in the order the trees were given.

        Returns:
            Each tree's outputs as ``(band, y, x)`` on that tree's grid.

        Raises:
            ValueError: A tree received no outputs.
        """
        rasters = []
        for tree_index, grid in enumerate(self.grids):
            if tree_index not in self.mergers:
                raise ValueError(
                    f"tree {tree_index} received no outputs; add one per chip"
                )
            halo = [(0, 0), *self.halos[tree_index]]
            values = self.mergers[tree_index].merge(extra_padding=halo, unpad=True)
            output = xr.DataArray(values, dims=("band", "y", "x"), name="output")
            rasters.append(output.gs.write_grid(grid))
        return rasters


def lay_grid(tree: xr.DataTree, resolution: float) -> GeoBox:
    """Lay a grid at `resolution` over the ground every group of a tree covers.

    Groups may sit on their own grids, such as optical at 10 m and radar at
    20 m; the grid spans their joint bounding box in their shared CRS.

    Args:
        tree: Tree whose groups each sit on a grid.
        resolution: Pixel size in the units of the groups' CRS.

    Returns:
        The covering grid.

    Raises:
        ValueError: A group sits on no grid, or the groups lie in different CRSs.
    """
    ground = None
    for name, raster in tree.gs.rasters.items():
        geobox = raster.gs.geobox
        if geobox is None:
            raise ValueError(f"group {name!r} sits on no grid to lay one over")
        if ground is not None and geobox.crs != ground.crs:
            raise ValueError(
                f"the groups lie in {ground.crs} and {geobox.crs}; one grid needs one CRS"
            )
        ground = geobox.extent if ground is None else ground | geobox.extent
    if ground is None:
        raise ValueError("the tree holds no groups to lay a grid over")
    return GeoBox.from_bbox(ground.boundingbox, ground.crs, resolution=resolution)


def lay_tiles(
    shape: tuple[int, int], size: int, overlap: int
) -> tuple[Tiler, list[tuple[int, int]]]:
    """Lay chips over a grid, with a halo around it when chips overlap.

    Args:
        shape: Grid height and width.
        size: Chip width and height in pixels.
        overlap: Pixels neighbouring chips share.

    Returns:
        The Tiler, and the halo pixels before and after `y`, then `x`.
    """
    tiler = Tiler(data_shape=shape, tile_shape=(size, size), overlap=overlap)
    halo = [(0, 0), (0, 0)]
    if overlap:
        padded_shape, padding = tiler.calculate_padding()
        tiler.recalculate(data_shape=padded_shape)
        halo = []
        for before, after in padding:
            halo.append((int(before), int(after)))
    return tiler, halo
