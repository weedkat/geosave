"""Reproduce selected cf-xarray adoption findings without modifying library data.

Run from the repository with:
    MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python docs/contributing/reviews/cf_xarray_smoke.py

These checks describe the audited installed behavior, including current gaps.
They are research evidence, not desired-behavior regression tests.
"""

import cf_xarray
import numpy as np
import xarray as xr
from dask.callbacks import Callback

import geosave.geodata  # noqa: F401 registers .gs
from geosave.geodata.attrs import Legend


def inspect_flags():
    """Compare declaration recognition, validation and pixel interpretation."""
    malformed = xr.DataArray(
        [0, 1],
        dims="pixel",
        attrs={"flag_values": [0, 1], "flag_meanings": "clear"},
    )
    assert malformed.cf.is_flag_variable
    try:
        _ = malformed.cf.flags
    except ValueError:
        print("is_flag_variable accepts mismatched lengths; flags rejects them")
    else:
        raise AssertionError("Recheck flag validation behavior")

    combined = xr.DataArray(
        np.arange(4, dtype="uint8"),
        dims="pixel",
        attrs={
            "flag_masks": [1, 3],
            "flag_values": [1, 1],
            "flag_meanings": "bit_set code_one",
        },
    ).chunk(pixel=2)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        flags = combined.cf.flags
    assert not tasks
    np.testing.assert_array_equal(flags.bit_set.compute(), [False, True, False, True])
    np.testing.assert_array_equal(flags.code_one.compute(), [False, True, False, False])
    assert not combined.cf.isin(["typo"]).compute().any().item()
    legend = Legend.from_attrs(combined.attrs)
    assert legend is not None and legend.classes is None
    print("CF combined flags work lazily; Legend retains their masked declarations")

    palette = xr.DataArray([0, 1], attrs={"color_map": '{"0":"#000000"}'})
    assert not palette.cf.is_flag_variable
    assert palette.gs.is_categorical
    print("Palette categorical protection is broader than CF flag detection")


def inspect_coordinates():
    """Separate coordinate discovery from dimension resolution."""
    ds = xr.Dataset(
        {"image": ("row", [1, 2])},
        coords={"lat": ("row", [10, 20], {"standard_name": "latitude"})},
    )
    assert ds.cf.coordinates["latitude"] == ["lat"]
    try:
        ds.cf.mean("latitude")
    except ValueError:
        print(
            "Auxiliary latitude is discoverable but not resolved as a reduction dimension"
        )
    else:
        raise AssertionError("Recheck auxiliary-coordinate dimension resolution")

    moved = ds.copy(deep=True)
    moved.lat.encoding["standard_name"] = moved.lat.attrs.pop("standard_name")
    assert "latitude" not in moved.cf.coordinates
    print("standard_name in encoding does not substitute for coordinate attrs")


def inspect_support_variables():
    """Check support-aware Dataset raster traversal."""
    ds = xr.Dataset(
        {
            "image": (("y", "x"), np.ones((2, 3))),
            "x_bounds": (("x", "nv"), [[0, 1], [1, 2], [2, 3]]),
        },
        coords={
            "y": ("y", [1.5, 0.5], {"axis": "Y"}),
            "x": ("x", [0.5, 1.5, 2.5], {"axis": "X", "bounds": "x_bounds"}),
        },
    )
    assert ds.cf.get_bounds("X").name == "x_bounds"
    assert ds.image.gs.spatial_dims == ("y", "x")
    assert ds.gs.spatial_dims == ("y", "x")
    print(
        "CF bounds resolve and remain separate from GeoSave raster variables"
    )


if __name__ == "__main__":
    print(f"cf-xarray {cf_xarray.__version__}; xarray {xr.__version__}")
    inspect_flags()
    inspect_coordinates()
    inspect_support_variables()
    print("All documented smoke observations reproduced")
