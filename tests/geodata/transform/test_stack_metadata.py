import pytest

from geosave_engine.geodata import stack
from geosave_engine.geodata.transform.combine import concat_time
from geosave_engine.geodata.transform.warp import reproject

from .conftest import build, geobox, grid_of


@pytest.mark.parametrize("operation", ["concat", "reproject"])
def test_rebuilt_stacks_preserve_root_attrs(operation):
    source = stack({"optical": build(geobox())})
    source.attrs = {"title": "training scene", "foreign": {"site": 7}}
    if operation == "concat":
        result = concat_time([source, source])
    else:
        coarse = grid_of(source).zoom_out(2)
        result = reproject(source, coarse)
        assert result.gs.geobox == coarse
    assert result.attrs == source.attrs
