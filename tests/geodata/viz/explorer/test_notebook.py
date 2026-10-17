"""Kernel comm and static/interactive backend regression checks."""

import sys
import pytest

pytestmark = pytest.mark.slow


def test_native_notebook_comm():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from jupyter_client import KernelManager

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("""
import sys
import os
os.environ.pop('VSCODE_CWD', None)
os.environ.pop('VSCODE_PID', None)
import geopandas as gpd
from shapely.geometry import Point
import geosave
import panel as pn
import bokeh
from IPython.utils.capture import capture_output
from panel.io.notebook import ipywidget

frame = gpd.GeoDataFrame({'value':[1]}, geometry=[Point(0,0)], crs=4326)
pn.config.comms = 'default'
with capture_output():
    import geosave.geodata.viz.plot
    viewer = frame.gs.explore(basemap='None')
with capture_output() as captured:
    bundle = viewer._repr_mimebundle_()
core = f'https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh.__version__}.min.js'
assert any(core in output.data.get('application/vnd.holoviews_load.v0+json', '')
           for output in captured.outputs)
if isinstance(bundle, tuple):
    bundle = bundle[0]
assert 'application/vnd.jupyter.widget-view+json' in bundle
widget = ipywidget(viewer)
assert widget.comm.target_name == 'jupyter.widget'
assert 'application/vnd.jupyter.widget-view+json' in widget._repr_mimebundle_()
assert 'panel.io.ipywidget' not in sys.modules
assert 'lonboard' not in sys.modules
viewer.close()
""")
        ]
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    nbclient.NotebookClient(notebook, km=manager, timeout=90).execute()
