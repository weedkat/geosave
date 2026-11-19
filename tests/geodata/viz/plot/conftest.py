"""Static figures render headless, and dated captions name a fixed place."""

import holoviews as hv
import matplotlib
import pytest

from geosave.geodata.core.region import GeoRegion
from geosave.geodata.utils.geocode import Place

matplotlib.use("agg")
hv.extension("matplotlib")


@pytest.fixture(autouse=True)
def matplotlib_backend():
    """Make matplotlib current again: an explorer test run earlier leaves bokeh."""
    hv.Store.set_current_backend("matplotlib")


@pytest.fixture(autouse=True)
def resolved_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GeoRegion, "locate", lambda _: Place(city="Test City"))
