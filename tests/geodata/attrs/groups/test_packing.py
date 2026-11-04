from __future__ import annotations

import pytest

from geosave.geodata import attrs


def test_a_zero_scale_is_refused() -> None:
    with pytest.raises(ValueError):
        attrs.Packing(scale_factor=0)
