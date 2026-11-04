"""Join the attrs of several xarray objects inside xarray's own concat and merge.

Pass `combine_attrs=attrs.combine_attrs`; xarray calls it once per variable,
coordinate and root with the attrs of every input at that place. Each call
joins one level, so a group is read off variables, coordinates or roots alike.

An input's attrs hold attr groups, whose keys mean one value together, and
plain keys, which belong to no group. A group is kept whole when its validated
values agree; a plain key is kept when every input states it alike. Missing
or differing attrs are dropped: half of a packing or a legend is a wrong one.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

from xarray.structure.merge import merge_attrs

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import DroppedAttrsWarning

from .groups.acdd import ACDD
from .groups.gdal import GDALBand
from .groups.legend import Legend
from .groups.nodata import Nodata
from .groups.packing import Packing
from .groups.quantity import Quantity
from .groups.spectral import Spectral
from .groups.stac import StacMetadata
from .groups.timespec import TimeSpec

type Attrs = Mapping[Any, Any]

GROUPS = (
    ACDD,
    GDALBand,
    Legend,
    Nodata,
    Packing,
    Quantity,
    Spectral,
    StacMetadata,
    TimeSpec,
)
GROUP_KEYS = frozenset().union(*(group.KEYS for group in GROUPS))


def combine_attrs(
    variable_attrs: Sequence[Attrs], context: object = None
) -> dict[str, Any]:
    """Join attrs, keeping what every input shares and dropping the rest.

    Args:
        variable_attrs: Attrs of the inputs joined, in order.
        context: Passed by xarray as a keyword to every `combine_attrs`
            callback. A join here does not use it.

    Returns:
        Attrs of the joined object.

    Warns:
        DroppedAttrsWarning: An attr differs between inputs or is missing from
            some, so the join leaves it out.

    Examples:
        >>> xr.concat(rasters, dim="time", combine_attrs=attrs.combine_attrs)
    """
    shared_attrs, dropped_keys = _intersect_attrs(variable_attrs)
    if dropped_keys:
        warnings.warn(
            f"joining drops attrs that differ between inputs or are missing from "
            f"some: {sorted(dropped_keys)}",
            DroppedAttrsWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return shared_attrs


def intersect(variable_attrs: Sequence[Attrs]) -> dict[str, Any]:
    """Return the attrs every input shares, each group kept whole or not at all.

    A group is compared as the value it validates to, so its aliases read
    alike. Unlike `combine_attrs`, nothing is warned about, as when bands of
    one raster are stacked.

    Args:
        variable_attrs: Nonempty attrs of the inputs to compare.

    Returns:
        New mapping of shared attrs, each group written once. Attrs missing
        from any input or different between inputs are omitted.

    Raises:
        ValueError: The sequence is empty or an attr group is invalid.

    Examples:
        >>> intersect([ds[name].attrs for name in ds.data_vars])
        {'units': '1'}
    """
    shared_attrs, _ = _intersect_attrs(variable_attrs)
    return shared_attrs


def _intersect_attrs(
    variable_attrs: Sequence[Attrs],
) -> tuple[dict[str, Any], set[str]]:
    """Return shared attrs and dropped keys, using each group's written aliases."""
    if not variable_attrs:
        raise ValueError("joining needs the attrs of at least one input")

    plain_attrs = []
    written_keys: set[str] = set()
    shared_keys = set(variable_attrs[0]) - GROUP_KEYS
    for attrs in variable_attrs:
        plain = {}
        for key, value in attrs.items():
            if key not in GROUP_KEYS:
                plain[key] = value
                written_keys.add(str(key))
        plain_attrs.append(plain)
        shared_keys.intersection_update(attrs)

    # xarray's drop_conflicts drops differing values but keeps keys that only
    # some inputs have, so keys missing from an input are removed afterwards.
    merged_attrs = cast("dict[Any, Any]", merge_attrs(plain_attrs, "drop_conflicts"))
    shared_attrs = {
        str(key): value for key, value in merged_attrs.items() if key in shared_keys
    }

    for group_type in GROUPS:
        groups = [group_type.from_attrs(attrs) for attrs in variable_attrs]
        for group in groups:
            if group is not None:
                written_keys.update(group.to_attrs())
        first_group = groups[0]
        if first_group is not None and all(
            group == first_group for group in groups[1:]
        ):
            shared_attrs.update(first_group.to_attrs())

    return shared_attrs, written_keys - shared_attrs.keys()
