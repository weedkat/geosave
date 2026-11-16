"""Cut chips from tensor samples and merge predictions back, as the raster pair does.

`TensorChipper` and `TensorMerger` serve a LightningModule step whose batch is
larger than the model's input, such as whole validation scenes. They share
`ChipLayout` and `create_chip_weights` with `RasterChipper` and `RasterMerger`,
so the step chips and merges exactly as a release's processor does on rasters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from affine import Affine
from odc.geo.geobox import GeoBox

from .layout import ChipLayout, ChipMode, lay_chip_offsets
from .merger import MergeWindow, create_chip_weights

if TYPE_CHECKING:
    import torch

# Padding modes as ChipLayout names them, after numpy.pad, by their torch names.
_TORCH_PADDING = {
    "reflect": "reflect",
    "edge": "replicate",
    "wrap": "circular",
    "constant": "constant",
}


class TensorChipper:
    """Lay chips over a batch of tensor samples; index to cut one chip sample.

    A chip sample keeps the batch's structure without its batch axis, as a
    Dataset returns a sample: the `pixels` entries are cut to the chip,
    `transform` becomes the chip's, and every other entry is the chip's own
    sample's, such as `crs` or a context tensor. Context read from the
    scene, such as `centroid`, stays the scene's. Collate chip samples with
    `torch.utils.data.default_collate`.

    Args:
        batch: Collated samples on one grid shape, with `transform` shaped
            `(batch, 6)` and `crs` shaped `(batch,)`, as
            `DataTree.gs.to_tensor(grid=True)` writes them.
        pixels: Paths of the tensors to cut, `/` joining nested keys, such as
            `"image"`. Each ends in `(y, x)` on the samples' grid.
        size: Chip height and width in pixels.
        stride: Pixels between chip starts; None takes `size`.
        mode: How chips meet the grid's edge, as `RasterChipper` takes it.
        constant_value: Fill for `"constant"`.

    Raises:
        ValueError: `pixels` is empty or its tensors differ in grid shape.

    Examples:
        >>> chipper = TensorChipper(batch, pixels=["image"], size=224, stride=192)
        >>> chips = default_collate([chipper[0], chipper[1]])
        >>> chips["image"].shape, chips["transform"].shape
        (torch.Size([2, 4, 224, 224]), torch.Size([2, 6]))
    """

    def __init__(
        self,
        batch: Mapping[str, Any],
        *,
        pixels: Sequence[str],
        size: int,
        stride: int | None = None,
        mode: ChipMode = "reflect",
        constant_value: int | float = 0,
    ) -> None:
        # Imported here so that importing geodata never pays for torch.
        import torch

        if not pixels:
            raise ValueError("name at least one pixel tensor to cut")
        self.batch = batch

        shape = tuple(_read_path(batch, pixels[0]).shape[-2:])
        geoboxes = []
        for transform, epsg in zip(batch["transform"], batch["crs"], strict=True):
            geoboxes.append(GeoBox(shape, Affine(*transform.tolist()), int(epsg)))
        self.layout = ChipLayout(geoboxes, size=size, stride=stride, mode=mode)

        # Pad once so every chip lies inside; "shift" and "drop" chips already do.
        stride = self.layout.stride
        rows = lay_chip_offsets(shape[0], size=size, stride=stride, mode=mode)
        columns = lay_chip_offsets(shape[1], size=size, stride=stride, mode=mode)
        bottom = 0
        right = 0
        if rows and columns:
            bottom = max(rows[-1] + size - shape[0], 0)
            right = max(columns[-1] + size - shape[1], 0)

        self.padded: dict[str, torch.Tensor] = {}
        for path in pixels:
            value = _read_path(batch, path)
            if tuple(value.shape[-2:]) != shape:
                raise ValueError(f"{path!r} is not on the grid of {pixels[0]!r}")
            # torch pads two axes only on 3D or 4D input, so leading axes are folded.
            folded = value.reshape(-1, *shape)
            padded = torch.nn.functional.pad(
                folded,
                (0, right, 0, bottom),
                mode=_TORCH_PADDING.get(mode, "constant"),
                value=constant_value if mode == "constant" else None,
            )
            self.padded[path] = padded.reshape(*value.shape[:-2], *padded.shape[-2:])

    def __len__(self) -> int:
        """Return the number of chips across every sample."""
        return len(self.layout)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Cut one chip sample, keeping the batch's structure without its batch axis.

        Raises:
            IndexError: The chip index is outside the chips.
        """
        return self._cut(self.batch, "", index)

    def _cut(self, value: Any, path: str, index: int) -> Any:
        """Return one entry of the chip sample at `path`."""
        # Imported here so that importing geodata never pays for torch.
        import torch

        if isinstance(value, Mapping):
            chip = {}
            for key, entry in value.items():
                chip[key] = self._cut(entry, f"{path}/{key}" if path else key, index)
            return chip

        location = self.layout.locate(index)
        if path in self.padded:
            rows = slice(location.top, location.top + self.layout.size)
            columns = slice(location.left, location.left + self.layout.size)
            return self.padded[path][location.raster_index, ..., rows, columns]
        if path == "transform":
            transform = self.layout.geobox(index).transform[:6]
            return torch.tensor(transform, dtype=torch.float64)
        return value[location.raster_index]


class TensorMerger:
    """Accumulate dense chip predictions as tensors and merge them per sample.

    The tensor counterpart of `RasterMerger`: the same layout places each
    chip, the same window weights overlaps, and the part of a chip past its
    sample's grid is discarded. Sums stay on the predictions' device.

    Args:
        layout: The layout the predicted chips were cut by, such as
            `TensorChipper.layout`. Every grid has the same shape.
        window: Taper weighting each chip; `"boxcar"` averages overlaps evenly.

    Raises:
        ValueError: The layout's grids differ in shape.

    Examples:
        >>> merger = TensorMerger(chipper.layout, window="hann")
        >>> for index in range(len(chipper)):
        ...     merger.add(index, logits_of(chipper[index]))
        >>> merger.merge().shape
        torch.Size([2, 3, 500, 500])
    """

    def __init__(self, layout: ChipLayout, *, window: MergeWindow = "boxcar") -> None:
        shapes = set()
        for geobox in layout.geoboxes:
            shapes.add(tuple(geobox.shape))
        if len(shapes) != 1:
            raise ValueError("every grid of a TensorMerger layout needs one shape")
        self.layout = layout
        self.window: MergeWindow = window
        self.sums: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.chip_weights: torch.Tensor | None = None

    def add(self, index: int, prediction: torch.Tensor) -> None:
        """Accumulate one chip's prediction onto its sample.

        Args:
            index: The chip's index in the layout.
            prediction: Values shaped `(band, size, size)`.

        Raises:
            IndexError: The chip index is outside the chips.
            ValueError: The prediction is not `(band, size, size)`, or its band
                count differs from earlier chips.
        """
        # Imported here so that importing geodata never pays for torch.
        import torch

        size = self.layout.size
        if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (size, size):
            raise ValueError(
                f"prediction must be shaped (band, {size}, {size}), "
                f"got {tuple(prediction.shape)}"
            )

        if self.sums is None or self.weights is None or self.chip_weights is None:
            height, width = self.layout.geoboxes[0].shape
            count = len(self.layout.geoboxes)
            device = prediction.device
            band_count = prediction.shape[0]
            self.sums = torch.zeros((count, band_count, height, width), device=device)
            self.weights = torch.zeros((count, height, width), device=device)
            weights = create_chip_weights(self.window, size)
            self.chip_weights = torch.from_numpy(weights).to(device)
        if prediction.shape[0] != self.sums.shape[1]:
            raise ValueError(
                f"prediction has {prediction.shape[0]} bands; earlier chips "
                f"had {self.sums.shape[1]}"
            )

        # The part of the chip on the grid; padding past the edge is dropped.
        location = self.layout.locate(index)
        height, width = self.sums.shape[-2:]
        bottom = min(location.top + size, height)
        right = min(location.left + size, width)
        rows = bottom - location.top
        columns = right - location.left

        weights = self.chip_weights[:rows, :columns]
        weighted = prediction[:, :rows, :columns].float() * weights
        scene = location.raster_index
        self.sums[scene, :, location.top : bottom, location.left : right] += weighted
        self.weights[scene, location.top : bottom, location.left : right] += weights

    def merge(self) -> torch.Tensor:
        """Return every sample's merged predictions, in the batch's layout.

        Returns:
            float32 tensor shaped `(batch, band, y, x)`; NaN where no chip landed.

        Raises:
            ValueError: No prediction was added.
        """
        # Imported here so that importing geodata never pays for torch.
        import torch

        if self.sums is None or self.weights is None:
            raise ValueError("no prediction was added")
        weights = self.weights[:, None]
        merged = self.sums / weights
        return torch.where(weights > 0, merged, torch.nan)


def _read_path(batch: Mapping[str, Any], path: str) -> Any:
    """Return the nested entry a `/`-joined key path names."""
    value: Any = batch
    for key in path.split("/"):
        value = value[key]
    return value
