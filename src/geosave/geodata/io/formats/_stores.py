"""Checks the Zarr and netCDF writers make before they touch a destination."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr
from upath import UPath


def check_store(
    raster: xr.Dataset | xr.DataTree,
    store: UPath,
    suffixes: tuple[str, ...],
    *,
    overwrite: bool,
    encoding: Mapping[Any, Any] | None,
) -> None:
    """Refuse a destination or encoding a store write could not honour.

    Args:
        raster: Dataset, or a tree written with its groups.
        store: The destination.
        suffixes: Suffixes the format's stores end in.
        overwrite: Allow an existing destination.
        encoding: Storage settings keyed by variable name.

    Raises:
        ValueError: The destination's suffix is not the format's.
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names a variable no group holds.
    """
    if store.suffix.lower() not in suffixes:
        raise ValueError(f"destination {store.name!r} must end in {suffixes}")
    if not overwrite and store.exists():
        raise FileExistsError(f"{store} exists; pass overwrite=True to replace it")
    nodes = list(raster.subtree) if isinstance(raster, xr.DataTree) else [raster]
    names = set()
    for node in nodes:
        names.update(node.variables)
    unknown = set(encoding or {}) - names
    if unknown:
        raise KeyError(f"encoding names no variable: {sorted(unknown, key=str)}")
