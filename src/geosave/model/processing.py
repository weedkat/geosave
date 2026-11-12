"""Reusable model input checks, variable selection, and tensor boundaries.

Selection sets band identity and order. Validation checks declared requirements
without changing values. Decoding applies stored packing via gs.mask_and_scale;
normalization standardizes physical values using training statistics. Encoding
materializes pixels as a tensor; restoration places predictions on their grid.
These are ordinary operations callable from Python or ModelSpec recipes.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import torch


def to_tensor(
    raster: xr.Dataset | xr.DataArray, *, dims: Sequence[str], dtype: str = "float32"
) -> torch.Tensor:
    """Materialize a raster in an explicitly declared dimension order.

    Args:
        raster: Native raster. Dataset variables become bands in their current order.
        dims: Every output dimension, including band for a Dataset.
        dtype: PyTorch dtype name.

    Returns:
        CPU tensor in the declared order. Conversion reads lazy pixels.

    Raises:
        ValueError: Dataset variables have different dimension sets. Broadcast
            explicitly before encoding if repeating a variable is intended.
    """
    import torch

    if isinstance(raster, xr.Dataset):
        raster = raster[list(raster.gs.variables)]
        dimensions = {
            frozenset(variable.dims) for variable in raster.data_vars.values()
        }
        if len(dimensions) != 1:
            raise ValueError(
                "tensor bands must have the same dimensions; broadcast explicitly first"
            )
        array = raster.to_array(dim="band")
    else:
        array = raster
    values = array.transpose(*dims).to_numpy()
    # Views may have negative strides or be read-only, which torch cannot safely use.
    return torch.as_tensor(np.array(values, copy=True), dtype=getattr(torch, dtype))


def encode_rasters(
    rasters: Sequence[xr.Dataset],
    *,
    bands: Sequence[str],
    dims: Sequence[str],
    dtype: str = "float32",
) -> tuple[torch.Tensor, ...]:
    """Select bands, decode packing, and materialize rasters together.

    Args:
        rasters: Ordered Datasets; locations and shapes may differ.
        bands: Variables in model band order.
        dims: Every output axis, including band. No dimensions are broadcast.
        dtype: PyTorch dtype name.

    Returns:
        One CPU tensor per raster, in input order. Lazy arrays share one Dask
        computation. The entire collection must fit in memory; callers choose
        its size. Use torch.stack explicitly when shapes are compatible.

    Raises:
        ValueError: Bands or dimensions are incompatible; the error identifies
            the raster position. All metadata is checked before reading pixels.
    """
    from dask.base import compute
    import torch

    tensor_dtype = getattr(torch, dtype)
    arrays = []
    for position, raster in enumerate(rasters):
        try:
            selected = select_variables(raster, names=bands)
            dimensions = set(dims) - {"band"}
            for name, variable in selected.data_vars.items():
                if set(variable.dims) != dimensions:
                    raise ValueError(
                        f"Variable {name!r} has dimensions {variable.dims}; "
                        f"expected the non-band dimensions of {tuple(dims)}. "
                        "Select observations or broadcast explicitly before encoding."
                    )

            decoded = selected.gs.mask_and_scale()
            array = decoded.to_array(dim="band").transpose(*dims)
            arrays.append(array.data)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot encode raster at position {position}: {error}"
            ) from error

    values = compute(*arrays)
    tensors = []
    for value in values:
        tensors.append(torch.as_tensor(np.array(value, copy=True), dtype=tensor_dtype))
    return tuple(tensors)


def normalize(
    value: torch.Tensor, *, mean: Sequence[float], std: Sequence[float], dim: int
) -> torch.Tensor:
    """Normalize one tensor axis with fixed statistics, preserving gradients.

    Args:
        value: Floating tensor on any device, batched or unbatched.
        mean: One mean per element of the selected axis.
        std: One finite positive standard deviation per axis element.
        dim: Axis to normalize; negative axes remain stable when batching.

    Returns:
        Tensor with the same shape, dtype, and device. Missing values are preserved.

    Raises:
        ValueError: Statistics or axis lengths are invalid.
        TypeError: The tensor is not floating point.
    """
    import torch

    if not value.is_floating_point():
        raise TypeError("normalize requires a floating tensor")
    if not -value.ndim <= dim < value.ndim:
        raise ValueError(
            f"normalization axis {dim} is absent from shape {tuple(value.shape)}"
        )
    if len(mean) != value.shape[dim] or len(std) != value.shape[dim]:
        raise ValueError("normalization statistics must match the selected axis")
    center = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
    scale = torch.as_tensor(std, dtype=value.dtype, device=value.device)
    if not bool(
        torch.isfinite(center).all() & torch.isfinite(scale).all() & (scale > 0).all()
    ):
        raise ValueError(
            "normalization needs finite means and positive finite standard deviations"
        )
    shape = [1] * value.ndim
    shape[dim] = len(mean)
    return (value - center.reshape(shape)) / scale.reshape(shape)


def to_raster(
    value: torch.Tensor | np.ndarray,
    *,
    reference: xr.Dataset | xr.DataArray,
    dims: Sequence[str],
    name: str,
) -> xr.DataArray:
    """Place model values on an unchanged reference grid.

    Args:
        value: Output values. Tensors are detached and copied to CPU.
        reference: Raster defining exact spatial placement, not output band meaning.
        dims: Output axes; spatial axes must be y and x.
        name: Output variable name.

    Returns:
        Native DataArray on the reference grid, without inherited input packing.
        Assign output units, class labels, and nodata explicitly in the recipe.

    Raises:
        ValueError: The output does not cover the reference grid exactly.
    """
    import torch

    reference.gs.validate()
    values = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    grid = reference.gs.geobox
    if grid is None or "y" not in dims or "x" not in dims:
        raise ValueError("output requires one reference grid and y/x dimensions")
    if len(dims) != values.ndim:
        raise ValueError("output dimensions must match its rank")
    if (values.shape[dims.index("y")], values.shape[dims.index("x")]) != tuple(
        grid.shape
    ):
        raise ValueError(
            "output shape differs from the reference grid; supply its actual grid explicitly"
        )
    from odc.geo.xr import wrap_xr

    placed = wrap_xr(values, grid, dims=tuple(dims), axis=dims.index("y"))
    return placed.rename(name)


def select_variables(raster: xr.Dataset, *, names: Sequence[str]) -> xr.Dataset:
    """Select Dataset variables in model band order without reading pixels.

    Args:
        raster: Dataset containing the requested variables.
        names: Nonempty, unique variable names in tensor band order.

    Returns:
        Dataset retaining coordinates, metadata, and lazy arrays.

    Raises:
        ValueError: Names are empty, repeated, or absent from the Dataset.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one variable, with no repeated names.")

    missing = [name for name in names if name not in raster.data_vars]
    if missing:
        raise ValueError(
            f"Cannot select variables {missing}; available variables are "
            f"{list(raster.data_vars)}. Rename variables or change the selection."
        )

    return raster[list(names)]


def restore_rasters(
    values: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    *,
    references: Sequence[xr.Dataset | xr.DataArray],
    dims: Sequence[str],
    name: str,
) -> tuple[xr.DataArray, ...]:
    """Place each prediction on its corresponding reference grid.

    Args:
        values: Ordered predictions, or a tensor/array with samples on axis zero.
        references: Original rasters in the same order as predictions.
        dims: Dimensions of one prediction, excluding the batch axis.
        name: Output variable name.

    Returns:
        One georeferenced DataArray per prediction. Tensor predictions are
        detached and moved to CPU. Input packing and band metadata are not copied.

    Raises:
        ValueError: Counts differ or a prediction does not fit its reference grid.
    """
    import torch

    if len(values) != len(references):
        raise ValueError(
            f"Received {len(values)} predictions for {len(references)} reference rasters. "
            "Keep one prediction per reference, in the same order."
        )

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    rasters = []
    for position, (value, reference) in enumerate(zip(values, references, strict=True)):
        try:
            raster = to_raster(value, reference=reference, dims=dims, name=name)
        except ValueError as error:
            raise ValueError(
                f"Cannot restore prediction at position {position}: {error}"
            ) from error
        rasters.append(raster)
    return tuple(rasters)


def validate_raster(
    raster: xr.Dataset,
    *,
    dims: Sequence[str] | None = None,
    sizes: dict[str, int] | None = None,
    units: dict[str, str] | None = None,
) -> xr.Dataset:
    """Check a model's Dataset requirements without reading or changing pixels.

    Args:
        raster: Dataset with one georeferenced affine grid.
        dims: Exact dimension order required for every variable, if supplied.
        sizes: Required lengths for named dimensions; other lengths are unrestricted.
        units: Required units attributes for named variables, compared literally.
            This checks metadata, not whether pixel values obey those units.

    Returns:
        The original Dataset. Select variables, observations, and decode packing
        explicitly before checking the requirements for the next operation.

    Raises:
        TypeError: Input is not a Dataset.
        ValueError: Raster structure or a declared requirement is not satisfied.
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            f"Expected an xarray Dataset, received {type(raster).__name__}. "
            "Convert a DataArray with to_dataset or select a Dataset from the DataTree."
        )

    raster.gs.validate()

    if dims is not None:
        expected = tuple(dims)
        for name in raster.gs.variables:
            variable = raster[name]
            if variable.dims != expected:
                raise ValueError(
                    f"Variable {name!r} has dimensions {variable.dims}; "
                    f"expected {expected}. Select observations or transpose "
                    "dimensions explicitly before this step."
                )

    for dim, expected_size in (sizes or {}).items():
        actual_size = raster.sizes.get(dim)
        if actual_size != expected_size:
            raise ValueError(
                f"Dimension {dim!r} has size {actual_size}; expected {expected_size}. "
                "Select the required observations or crop/resample explicitly."
            )

    for name, expected_units in (units or {}).items():
        if name not in raster.data_vars:
            raise ValueError(
                f"Cannot check units for missing variable {name!r}; "
                f"available variables are {list(raster.data_vars)}."
            )

        actual_units = raster[name].attrs.get("units")
        if actual_units != expected_units:
            raise ValueError(
                f"Variable {name!r} declares units {actual_units!r}; "
                f"expected {expected_units!r}. Check the source metadata and "
                "convert values explicitly if their units differ."
            )

    return raster
