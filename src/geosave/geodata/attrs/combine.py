"""Use xarray's attr policies and warn when a policy drops keys."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from xarray.structure.merge import merge_attrs

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import DroppedAttrsWarning

if TYPE_CHECKING:
    from xarray.core.types import CombineAttrsOptions


def combine_attrs(
    variable_attrs: Sequence[Mapping[Any, Any]],
    context: object = None,
    *,
    policy: CombineAttrsOptions = "drop_conflicts",
) -> dict[str, Any]:
    """Combine attrs with a native xarray policy, warning about dropped keys.

    Args:
        variable_attrs: Attr mappings supplied by xarray or a metadata reader.
        context: Callback context supplied by xarray and forwarded unchanged.
        policy: Native xarray policy or callable. By default, conflicting keys
            are dropped and keys present in only one input are retained.

    Returns:
        Combined attrs, without group validation or alias normalization.

    Warns:
        DroppedAttrsWarning: The policy omits a key present in an input.

    Examples:
        >>> xr.concat(rasters, dim="time", combine_attrs=attrs.combine_attrs)
    """
    # Our callback already reports its own omissions. Calling it as a policy
    # must not add another copy of the same diagnostic.
    if callable(policy) and (
        policy is combine_attrs
        or isinstance(policy, partial)
        and policy.func is combine_attrs
    ):
        return policy(variable_attrs, context=context)

    combined = cast(
        "dict[str, Any]", merge_attrs(variable_attrs, policy, context=context) or {}
    )
    keys = set()
    for attributes in variable_attrs:
        keys.update(attributes)

    dropped = keys - combined.keys()
    if dropped:
        warnings.warn(
            f"joining drops attrs: {sorted(dropped)}",
            DroppedAttrsWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return combined
