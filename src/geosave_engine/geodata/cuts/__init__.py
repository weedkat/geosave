"""Cut an item table into the windows a model reads: frames in time, chips in space.

Every cut takes a table and returns a table, opening no file. `stacks` lists one
window per anchor item, joined by the items covering it; `frames` and `chips`
split windows; a row reads itself with `row.gs.to_stack()`; `merge` puts chip
outputs back together.

Examples:
    >>> windows = cuts.stacks(gs.open_vector("data/catalog.parquet"), on="label")
    >>> windows = cuts.chips(cuts.frames(windows, 4, tolerance="10D"), 224)
    >>> chip = windows.iloc[0].gs.to_stack()
"""

from .sample import GeoSample
from .chips import chips, layout
from .frames import frames
from .merge import merge
from .stacks import stacks

__all__ = [
    "GeoSample",
    "chips",
    "frames",
    "layout",
    "merge",
    "stacks",
]
