"""Which Item each loaded `time` step is."""

import numpy as np
import pytest
import xarray as xr

from geosave.geodata.stac.stamps import LoadSource
from tests.geodata.stac.stamps.conftest import scene, stepped


def test_steps_match_items_in_step_order() -> None:
    first, second = scene("a", 1), scene("b", 2)

    assert LoadSource([first, second]).match_steps(stepped(second, first)) == [
        second,
        first,
    ]


def test_a_source_is_frozen() -> None:
    source = LoadSource([])

    with pytest.raises(AttributeError):
        source.items = []


def test_fused_steps_match_no_items() -> None:
    first, second = scene("a", 1), scene("b", 2)
    fused = stepped(first)

    assert LoadSource([first, second]).match_steps(fused) is None


def test_a_timeless_load_matches_no_items() -> None:
    loaded = xr.Dataset({"red": (("y", "x"), [[1]])})

    assert LoadSource([scene()]).match_steps(loaded) is None


def test_items_sharing_an_instant_match_no_steps() -> None:
    first, second = scene("a", 1), scene("b", 1)
    loaded = stepped(first, second).assign_coords(
        time=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
    )

    assert LoadSource([first, second]).match_steps(loaded) is None
