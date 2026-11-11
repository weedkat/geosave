"""Read explicitly paired image and label rasters for this project."""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

import geosave as gs
from geosave.model.spec import ModelSpec

from .data import Sample


class RasterDataset(Dataset[Sample]):
    """Read one selected observation and its aligned class labels per index.

    This project's recipe accepts `raster` and prepares native reflectance.
    Files must already cover the desired chip; no sampling or alignment is implicit.

    Args:
        images: Image files, in sample order. Paths and URLs use GeoSave I/O.
        targets: Single-band label files in the corresponding order.
        spec: Processing recipe, or its YAML file.
        num_classes: Number of classes, with IDs from zero to this value minus one.
        ignore_index: Label excluded from loss and metrics; also used for nodata.

    Raises:
        ValueError: Lists differ in length, classes are invalid, or a read finds
            mismatched grids, invalid labels, or nonfinite image inputs.
    """

    def __init__(
        self,
        images: list[str],
        targets: list[str],
        spec: str | Path | ModelSpec,
        num_classes: int,
        ignore_index: int = 255,
    ) -> None:
        if len(images) != len(targets):
            raise ValueError("images and targets must contain the same number of files")
        if num_classes < 1 or 0 <= ignore_index < num_classes:
            raise ValueError(
                "num_classes must be positive and ignore_index outside class IDs"
            )
        self.images = list(images)
        self.targets = list(targets)
        self.spec = spec if isinstance(spec, ModelSpec) else ModelSpec.load(spec)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        """Return the number of image/label pairs."""
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        """Read and validate one pair, returning tensors before shared transforms."""
        with (
            gs.open_raster(self.images[index]) as image,
            gs.open_raster(self.targets[index]) as labels,
        ):
            if not isinstance(image, xr.Dataset) or not isinstance(labels, xr.Dataset):
                raise ValueError(
                    "select one raster from each hierarchy before training"
                )
            prepared = self.spec.run(
                entry="from_raster", output=["prepared", "encoded"], raster=image
            )
            reference = prepared["prepared"]
            if labels.gs.geobox is None or labels.gs.geobox != reference.gs.geobox:
                raise ValueError(
                    "image and target must have the same grid; align explicitly"
                )
            if len(labels.data_vars) != 1:
                raise ValueError("target must contain one class band")
            label = next(iter(labels.data_vars.values()))
            if label.dims != ("y", "x"):
                raise ValueError(
                    "target dimensions must be (y, x); select one observation"
                )
            pixels = label.gs.mask_and_scale().to_numpy()
            missing = np.isnan(pixels) | (pixels == self.ignore_index)
            known = pixels[~missing]
            if (
                not np.isfinite(known).all()
                or not np.equal(known, np.floor(known)).all()
            ):
                raise ValueError(
                    "target must contain finite integer class IDs or nodata"
                )
            if np.any((known < 0) | (known >= self.num_classes)):
                raise ValueError("target class IDs must be within num_classes")
            target = torch.from_numpy(
                np.where(missing, self.ignore_index, pixels).astype("int64")
            )
            inputs = {"image": prepared["encoded"]}
            for name in inputs:
                if not torch.isfinite(inputs[name]).all():
                    raise ValueError(
                        f"{name} must be finite; declare a nodata policy in the recipe"
                    )
                if inputs[name].shape[-2:] != target.shape:
                    raise ValueError("image tensor and target must cover the same grid")
            return {"inputs": inputs, "target": target}
