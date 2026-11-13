"""Registry of stamps by name, filled by `@register_stamp` where each is defined."""

from collections.abc import Callable, Sequence

import xarray as xr

from .source import LoadSource, Stamp

STAMPS: dict[str, Stamp] = {}


def register_stamp(name: str) -> Callable[[Stamp], Stamp]:
    """Make a stamp selectable by name in `load_dataset(stamps=...)`.

    Examples:
        >>> @register_stamp("dw_source")
        ... def stamp_dw_source(loaded, source):
        ...     return loaded.gs.write_stac_property("dw:source", "experts")

    Raises:
        ValueError: The name is taken.
    """

    def register(stamp: Stamp) -> Stamp:
        if name in STAMPS:
            raise ValueError(f"stamp {name!r} is already registered")
        STAMPS[name] = stamp
        return stamp

    return register


def check_stamps(names: Sequence[str]) -> None:
    """Refuse names no stamp is registered under, before anything is loaded.

    Raises:
        KeyError: A name is unknown; the message lists the registered names.
    """
    unknown = []
    for name in names:
        if name not in STAMPS:
            unknown.append(name)
    if unknown:
        raise KeyError(f"unknown stamps {unknown}; registered: {sorted(STAMPS)}")


def run_stamps(
    loaded: xr.Dataset, source: LoadSource, names: Sequence[str]
) -> xr.Dataset:
    """Apply the named stamps in order; a later stamp overrides an earlier one.

    Raises:
        KeyError: A name is unknown; no stamp runs.
    """
    check_stamps(names)

    for name in names:
        loaded = STAMPS[name](loaded, source)
    return loaded
