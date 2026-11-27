"""Batch DataLoader samples that hold xarray objects."""

from collections.abc import Sequence
from typing import Any

import xarray as xr


def _keep_list(values: Sequence[Any], *, collate_fn_map: Any = None) -> list[Any]:
    """Return a batch's values as a list, unstacked."""
    return list(values)


def collate(batch: Sequence[Any]) -> Any:
    """Batch samples as PyTorch's `default_collate` does, keeping xarray objects as lists.

    Tensors, arrays and numbers are stacked, and tuples and mappings are
    batched field by field, as `default_collate` batches them. A DataArray,
    Dataset or DataTree keeps its own grid and size: its field becomes a list
    of the batch's objects, in sample order.

    Args:
        batch: The samples a DataLoader drew for one batch.

    Returns:
        The batch, structured like one sample.

    Examples:
        >>> loader = DataLoader(scenes, batch_size=4, collate_fn=gs.collate)
        >>> images, labels = next(iter(loader))
        >>> len(images)
        4
    """
    # Imported here so that importing geodata does not load torch.
    from torch.utils.data._utils.collate import collate as collate_fields
    from torch.utils.data._utils.collate import default_collate_fn_map

    rules = default_collate_fn_map | {
        xr.DataArray: _keep_list,
        xr.Dataset: _keep_list,
        xr.DataTree: _keep_list,
    }
    return collate_fields(batch, collate_fn_map=rules)
