import numpy as np
import pytest

from geosave.geodata import tree
from geosave.geodata.transform.temporal import concat
from geosave.geodata.transform.nodata import mask, to_nan
from geosave.geodata.transform.warp import reproject

from .conftest import build, geobox, grid_of


@pytest.mark.parametrize("operation", ["concat", "reproject", "mask", "to_nan"])
def test_rebuilt_trees_preserve_root_attrs(operation):
    source = tree({"optical": build(geobox())})
    source.attrs = {"title": "training scene", "foreign": {"site": 7}}
    if operation == "concat":
        result = concat([source, source])
    elif operation == "reproject":
        coarse = grid_of(source).zoom_out(2)
        result = reproject(source, coarse)
        assert result.gs.geobox == coarse
    elif operation == "mask":
        optical = source.gs.rasters["optical"]
        valid = np.ones((optical.sizes["y"], optical.sizes["x"]), bool)
        result = mask(source, valid, fill=0)
    else:
        result = to_nan(source)
    assert result.attrs == source.attrs
