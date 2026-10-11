"""Name the data variables a raster call refers to."""

from collections.abc import Sequence

import xarray as xr


def variable_names(raster: xr.Dataset, target: str | Sequence[str] | None) -> list[str]:
    """Return the data variable names `target` refers to, every one when None.

    Args:
        raster: Dataset holding the variables.
        target: One variable name, several, or None for every data variable.

    Returns:
        Variable names, in the order `target` gives or else in Dataset order.

    Raises:
        KeyError: A name is not a data variable of the raster.
    """
    carried = [str(name) for name in raster.data_vars]
    if target is None:
        return carried
    if isinstance(target, str):
        names = [target]
    else:
        names = list(target)

    missing = [name for name in names if name not in carried]
    if missing:
        raise KeyError(
            f"{missing} are not data variables of this raster; it carries {carried}"
        )
    return names
