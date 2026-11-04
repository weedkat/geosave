"""Static figures render headless, and dated captions name a fixed place."""

import holoviews as hv
import matplotlib
import pytest

from geosave.geodata.core.anchor import GeoAnchor
from geosave.geodata.utils.geocode import Place

matplotlib.use("agg")
hv.extension("matplotlib")


@pytest.fixture(autouse=True)
def resolved_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GeoAnchor, "locate", lambda _: Place(city="Test City"))
