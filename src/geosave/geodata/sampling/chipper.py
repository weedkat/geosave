"""Cut the chips of a layout from prepared rasters, lazily."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from geosave.geodata.core.raster.datatree import same_geobox
from geosave.geodata.core.raster.transform.extent import Padding

from .layout import ChipLayout, ChipMode


class RasterChipper:
    """Lay fixed-size chips over prepared rasters; index to cut one lazily.

    Each raster is a tree of nodes on one GeoBox, such as
    `{"image": ..., "label": ...}`; rasters may differ in grid and size.
    `layout` says where each chip sits; construction reads only the grids.
    `RasterMerger` places predictions back by chip index.

    Args:
        rasters: Prepared rasters, such as the windows of
            `tree.gs.sample(...)` or several scenes.
        size: Chip height and width in pixels.
        stride: Pixels between chip starts; None takes `size`, so chips do
            not overlap.
        mode: How chips meet a raster's edge. `"reflect"`, `"edge"`, `"wrap"`
            and `"constant"` let the last chips reach past the edge and fill
            them as `numpy.pad` fills the whole raster; a reflection keeps
            texture continuous at the border. `"shift"` moves the last chips
            inward, flush with the edge, and invents no pixels; `"drop"`
            leaves incomplete chips out.
        constant_value: Fill for `"constant"`.

    Raises:
        RasterValidationError: A raster holds no georeferenced node, or its
            nodes sit on several GeoBoxes.
        ValueError: `size` or `stride` is below 1, `mode` is unknown, or
            `"shift"` meets a raster smaller than one chip.

    Examples:
        >>> tree = xr.DataTree.from_dict({"image": scene, "label": labels})
        >>> chipper = RasterChipper([tree], size=256, stride=224)
        >>> chipper.layout.locate(1)
        ChipLocation(raster_index=0, top=0, left=224)
        >>> chipper[1].gs.to_tensor()["image"].shape
        torch.Size([4, 256, 256])
    """

    def __init__(
        self,
        rasters: Sequence[xr.DataTree],
        *,
        size: int,
        stride: int | None = None,
        mode: ChipMode = "reflect",
        constant_value: int | float = 0,
    ) -> None:
        self.rasters = tuple(rasters)
        self.constant_value = constant_value

        geoboxes = []
        for raster in self.rasters:
            raster.gs.validate(same_geobox)
            geobox = raster.gs.geobox
            if geobox is None:
                raise ValueError("every raster needs at least one raster node")
            geoboxes.append(geobox)

        self.layout = ChipLayout(geoboxes, size=size, stride=stride, mode=mode)

    def __len__(self) -> int:
        """Return the number of chips across every raster."""
        return len(self.layout)

    def __getitem__(self, index: int) -> xr.DataTree:
        """Cut one lazy chip: every node of its raster on the chip's grid.

        Raises:
            IndexError: The chip index is outside the chips.
        """
        raster = self.rasters[self.layout.locate(index).raster_index]
        # "shift" and "drop" chips lie inside the raster, where nothing is filled.
        pad_mode: Padding = "constant"
        if self.layout.mode not in ("shift", "drop"):
            pad_mode = self.layout.mode
        return raster.gs.crop_to_grid(
            self.layout.geobox(index),
            mode=pad_mode,
            constant_value=self.constant_value,
        )
