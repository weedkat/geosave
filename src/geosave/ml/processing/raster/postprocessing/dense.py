"""Decode merged dense output into a class map or a raster of values."""

import numpy as np
import xarray as xr

from geosave.ml.processing.raster.postprocessing.base import (
    POSTPROCESSORS,
    RasterPostprocessor,
)

# Stored value of pixels no chip predicted.
NO_PREDICTION = 255


def copy_scalar_coords(product: xr.Dataset, sample: xr.DataTree) -> xr.Dataset:
    """Give a product the scalar coordinates of its sample, such as its `time`.

    A predictor rebuilds coordinates from the grid, so what identifies the
    sample, its acquisition time, is carried over here.
    """
    for node in sample.children.values():
        for name, coord in node.coords.items():
            if coord.ndim == 0 and name not in product.coords:
                product = product.assign_coords({name: coord})
    return product


def mask_product(product: xr.Dataset, sample: xr.DataTree, mask: str) -> xr.Dataset:
    """Set a product to nodata wherever a sample's mask node is false.

    Args:
        product: Raster product whose variables state their nodata.
        sample: Sample holding the mask on the product's grid.
        mask: Node of one boolean variable, true where the product is kept.

    Raises:
        KeyError: The sample has no such node.
        ValueError: The node holds another number of variables than one.
    """
    node = sample[mask].to_dataset()
    if len(node.data_vars) != 1:
        raise ValueError(f"mask node {mask!r} must hold one variable")
    keep = node[next(iter(node.data_vars))].compute()

    masked = product.copy()
    for name, variable in product.data_vars.items():
        masked[name] = variable.where(keep, variable.gs.nodata)
    return masked


@POSTPROCESSORS.register("segmentation")
class SegmentationPostprocessor(RasterPostprocessor[xr.DataArray, xr.Dataset]):
    """Decode merged logits into a class map that names its classes.

    Args:
        classes: Class names; a class id is its position.
        colors: Hex colour of each class.
        mask: Sample node, true where the class map is kept; None keeps all.
    """

    def __init__(self, classes: list[str], colors: list[str], mask: str | None = None):
        self.classes = classes
        self.colors = colors
        self.mask = mask

    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset:
        """Take the strongest class of merged logits and build the class map.

        Args:
            output: Merged float32 `(band, y, x)` logits, one band per class.
            sample: Sample the logits were predicted over; holds the `mask`
                node and the scalar coordinates.

        Returns:
            Dataset of one variable, `class_id` (uint8, `(y, x)`)
            on the logits' grid, each value a position in `classes`. It
            states the class names and colours as flags and `NO_PREDICTION`
            as nodata, which fills pixels whose logits are missing or whose
            mask is false, and carries the sample's scalar coordinates.
        """
        logits = output.values

        # Read band by band: no copy of the whole logits is made to find gaps.
        missing = np.zeros(logits.shape[1:], dtype=bool)
        for band in logits:
            missing |= np.isnan(band)
        class_ids = logits.argmax(axis=0).astype("uint8")
        class_ids[missing] = NO_PREDICTION

        # One band's view lends its grid and coordinates; the logits are not copied.
        grid = output.isel(band=0, drop=True)
        class_map = grid.copy(data=class_ids).to_dataset(name="class_id")

        classes = dict(enumerate(self.classes))
        colors = dict(enumerate(self.colors))
        class_map = class_map.gs.write_flags(classes, colors)
        class_map = class_map.gs.write_nodata(NO_PREDICTION)

        product = copy_scalar_coords(class_map, sample)
        if self.mask is not None:
            product = mask_product(product, sample, self.mask)
        return product


@POSTPROCESSORS.register("regression")
class RegressionPostprocessor(RasterPostprocessor[xr.DataArray, xr.Dataset]):
    """Name merged per-pixel values as the variables a model estimates.

    Args:
        names: Variable name of each output band, in model order.
        units: Unit of the values, such as `"Mg/ha"`; None states none.
        mask: Sample node, true where the values are kept; None keeps all.

    Examples:
        >>> postprocessor = RegressionPostprocessor(names=["agbd"], units="Mg/ha")
        >>> postprocessor(output, sample)["agbd"].dtype
        dtype('float32')
    """

    def __init__(
        self, names: list[str], units: str | None = None, mask: str | None = None
    ):
        self.names = names
        self.units = units
        self.mask = mask

    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset:
        """Split merged values into one named variable per band.

        Args:
            output: Merged float32 `(band, y, x)` values, one band per name.
            sample: Sample the values were predicted over; holds the `mask`
                node and the scalar coordinates.

        Returns:
            Dataset of one float32 `(y, x)` variable per entry of
            `names`, on the values' grid, each stating `units` and NaN as
            nodata, which fills pixels no chip predicted or whose mask is
            false; it carries the sample's scalar coordinates.

        Raises:
            ValueError: The output has another number of bands than `names`.
        """
        if output.sizes["band"] != len(self.names):
            raise ValueError(
                f"the model returned {output.sizes['band']} bands for "
                f"{len(self.names)} names {self.names}"
            )

        values = xr.Dataset()
        for index, name in enumerate(self.names):
            values[name] = output.isel(band=index, drop=True)
            if self.units is not None:
                values[name].attrs["units"] = self.units
        values = values.gs.write_nodata(np.nan)

        product = copy_scalar_coords(values, sample)
        if self.mask is not None:
            product = mask_product(product, sample, self.mask)
        return product
