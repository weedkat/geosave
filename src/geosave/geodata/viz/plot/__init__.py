"""Static figures behind `.gs.plot`, drawn through hvPlot and rendered with matplotlib.

`hv.save(figure, "scene.png")` writes one directly, and figures compose with
`+` (side by side) and `*` (overlaid). Class maps share a legend to the right
of the figure, with each name and colour shown once.
"""

import holoviews as hv

from .raster import draw_raster
from .vector import draw_vector

hv.extension("matplotlib")

__all__ = ["draw_raster", "draw_vector"]
