"""Read explicitly paired image and label rasters for this project."""

from typing import Any, cast

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

import geosave as gs
from geosave.ml.factory import BuildSpec
from geosave.ml.processing.raster.preprocessing.base import PREPROCESSORS

from .data import Sample


class RasterDataset(Dataset[Sample]):
    """Read one selected observation and its aligned class labels per index.

    The preprocessor prepares each image raster, as the release's processor will.
    Files must already cover the desired chip; no sampling or alignment is implicit.

    Args:
        images: Image files, in sample order. Paths and URLs use GeoSave I/O.
        targets: Single-band label files in the corresponding order.
        preprocessor: Preprocessor by registered `name` or `class_path`, with
            `init_args`; its `preprocess` runs on every raster read.
        num_classes: Number of classes, with IDs from zero to this value minus one.
        ignore_index: Label excluded from loss and metrics; also used for nodata.
        context: Names in `geosave.geodata.TENSOR_CONTEXT` the model reads
            beside the pixels, such as `["centroid", "year", "day_of_year"]`.

    Raises:
        ValueError: Lists differ in length, classes are invalid, or a read finds
            mismatched grids, invalid labels, or nonfinite image inputs.
    """

    def __init__(
        self,
        images: list[str],
        targets: list[str],
        preprocessor: dict[str, Any],
        num_classes: int,
        ignore_index: int = 255,
        context: list[str] | None = None,
    ) -> None:
        if len(images) != len(targets):
            raise ValueError("images and targets must contain the same number of files")
        if num_classes < 1 or 0 <= ignore_index < num_classes:
            raise ValueError(
                "num_classes must be positive and ignore_index outside class IDs"
            )
        self.images = list(images)
        self.targets = list(targets)
        self.preprocessor = PREPROCESSORS.build(BuildSpec.model_validate(preprocessor))
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.context = context or []

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
            sources = xr.DataTree.from_dict({"image": image})
            reference = self.preprocessor(sources)["image"].to_dataset()
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
            # The grid lets the Module chip whole scenes as the release does.
            pixels, context = reference.gs.to_tensor(
                dtype="float32", grid=True, context=self.context
            )
            sample = {"image": pixels, **context}
            if not torch.isfinite(sample["image"]).all():
                raise ValueError(
                    "image must be finite; declare a nodata policy in the preprocessor"
                )
            if sample["image"].shape[-2:] != target.shape:
                raise ValueError("image tensor and target must cover the same grid")
            sample["target"] = target
            return cast(Sample, sample)
