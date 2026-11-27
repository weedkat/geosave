"""Mark property variables on the Dataset that ODC returned."""

import numpy as np
import xarray as xr

from geosave.geodata.stac.stamps.properties import stamp_properties
from tests.geodata.stac.stamps.conftest import scene


def test_mark_loaded_properties_and_preserve_the_raster():
    item = scene()
    item.properties.update({"eo:cloud_cover": 10.0, "platform": "demo"})
    ds = xr.Dataset(
        {
            "eo_cloud_cover": ("time", [10.0], {"units": "1"}),
            "red": (("time", "y", "x"), [[[2]]]),
        },
        coords={"time": [np.datetime64("2025-01-01")]},
    )
    out = stamp_properties(ds, [item])
    assert list(out.data_vars) == ["red"]
    assert out.gs.stac_properties == {"eo:cloud_cover": "eo_cloud_cover"}
    assert out.eo_cloud_cover.attrs == {"units": "1", "stac_property": "eo:cloud_cover"}
    assert "eo_cloud_cover" in ds.data_vars


def test_mark_a_custom_odc_output_name():
    ds = xr.Dataset({"cloud": ("time", [10.0])})
    out = stamp_properties(ds, [scene()], names={"cloud": "eo:cloud_cover"})
    assert out.gs.stac_properties == {"eo:cloud_cover": "cloud"}
