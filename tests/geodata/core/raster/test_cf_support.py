"""CF support survives pixel operations without becoming a raster band."""

import dask.array
import numpy as np
import pytest
import xarray as xr

from tests.geodata.conftest import build_raster


def bounded():
    scene = build_raster().chunk()
    scene["x_bounds"] = (
        ("x", "vertices"),
        np.column_stack([scene.x.values - 0.5, scene.x.values + 0.5]),
    )
    scene.x.attrs["bounds"] = "x_bounds"
    return scene


def test_unknown_nonspatial_data_is_not_silently_discarded():
    scene = build_raster().assign(unknown=42)
    assert "unknown" in scene.gs.raster_variables
    with pytest.raises(ValueError):
        scene.gs.validate()


def test_mapping_data_variable_is_not_a_band_and_keeps_georeferencing(tmp_path):
    from geosave.geodata.io import open_raster

    scene = build_raster().reset_coords("spatial_ref")
    for name in ("red", "nir"):
        scene[name].attrs["grid_mapping"] = "spatial_ref"
    scene.gs.validate(georeferenced=True)
    masked = scene.gs.mask(np.ones((2, 2), dtype=bool), fill=0)
    assert masked.gs.geobox == scene.gs.geobox
    assert "spatial_ref" in masked.data_vars
    assert scene.gs.geobox.crs is not None
    assert scene.gs.to_dataarray().gs.geobox == scene.gs.geobox
    path = tmp_path / "mapping.tif"
    from geosave.geodata.io.formats.geotiff import write_cog

    write_cog(scene, path)
    with open_raster(path) as restored:
        assert restored.gs.variables == ("red", "nir")
        assert restored.gs.geobox == scene.gs.geobox


def test_mask_updates_ancillary_raster_as_data_not_a_conflicting_coordinate():
    scene = build_raster()
    scene.red.attrs["ancillary_variables"] = "nir"
    result = scene.gs.mask(scene.red > 1000, fill=0)
    assert set(result.data_vars) == {"red", "nir"}
    np.testing.assert_array_equal(result.red.values, result.nir.values)
    assert result.red.attrs["ancillary_variables"] == "nir"


def test_spatial_operations_preserve_independent_coordinates_and_mapping():
    scene = build_raster().reset_coords("spatial_ref")
    for name in ("red", "nir"):
        scene[name].attrs["grid_mapping"] = "spatial_ref"
    scene = scene.assign_coords(levels=("level", [1, 2, 3]))
    grid = scene.gs.geobox
    for result in (
        scene.gs.write_grid(grid),
        scene.gs.reproject(grid.to_crs("EPSG:3857")),
        scene.gs.read_pixels(grid),
    ):
        xr.testing.assert_identical(result.levels.variable, scene.levels.variable)
        assert result.gs.geobox.crs is not None


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_tree_bounds_roundtrip_as_start_and_end(tmp_path, suffix):
    from geosave.geodata.io import open_raster

    time = np.array(["2025-01-15"], dtype="datetime64[ns]")
    root = xr.Dataset(
        {
            "cells": (
                ("time", "vertices"),
                np.array([["2025-01-01", "2025-02-01"]], dtype="datetime64[ns]"),
            )
        },
        coords={"time": ("time", time, {"standard_name": "time", "bounds": "cells"})},
    )
    tree = xr.DataTree.from_dict(
        {".": root, "image": build_raster().expand_dims(time=time)}
    )
    path = tmp_path / f"tree.{suffix}"
    if suffix == "nc":
        tree.gs.to_netcdf(path)
    else:
        tree.gs.to_zarr(path)
    with open_raster(path) as restored:
        assert list(restored.gs.datasets) == ["image"]
        assert restored.gs.timespan[0].day == 1
        image = restored.gs.datasets["image"]
        # Stored CF bounds come back as the cell coordinates every reader uses.
        assert "cells" not in image.variables
        np.testing.assert_array_equal(
            image.start_datetime.values, root.cells.values[:, 0]
        )
        np.testing.assert_array_equal(
            image.end_datetime.values, root.cells.values[:, 1]
        )


def test_thinned_cf_window_keeps_inferred_grid_on_singleton_axes():
    from pyproj import CRS

    scene = xr.Dataset(
        {
            "red": (
                ("y", "x"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                {"grid_mapping": "crs"},
            ),
            "crs": ((), 0, CRS.from_epsg(4326).to_cf()),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    target = scene.gs.geobox.zoom_out(2)
    pixels = scene.gs.read_pixels(target)
    assert pixels.gs.geobox == target
    assert pixels.red.shape == (1, 1)
    assert np.isfinite(pixels.red.values).all()


def test_reprojection_keeps_foreign_axis_names_and_coherent_mapping():
    from pyproj import CRS

    scene = xr.Dataset(
        {
            "red": (
                ("latitude", "longitude"),
                np.ones((2, 2)),
                {"grid_mapping": "crs"},
            ),
            "crs": ((), 0, CRS.from_epsg(4326).to_cf()),
        },
        coords={"latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    target = scene.gs.geobox.to_crs("EPSG:3857")
    warped = scene.gs.reproject(target)
    assert warped.red.dims == ("latitude", "longitude")
    assert warped.gs.geobox == target
    assert "grid_mapping" not in warped.red.attrs
    assert warped.red.encoding["grid_mapping"] in warped.coords
    sampled = scene.gs.read_pixels(target)
    assert "grid_mapping" not in sampled.red.attrs
    assert sampled.red.encoding["grid_mapping"] in sampled.coords


def test_cf_temporal_name_drives_cells_and_stac_stamping():
    from datetime import datetime, timezone
    import pystac
    from geosave.geodata.stac.raster import stamp_item_time, stamp_item_properties

    scene = build_raster().expand_dims(
        date=np.array(["2025-01-10", "2025-01-20"], dtype="datetime64[ns]")
    )
    scene.date.attrs.update(standard_name="time", axis="T")
    assert scene.gs.time_coordinate.name == "date"
    item = pystac.Item(
        "sample",
        None,
        None,
        datetime(2025, 1, 10, tzinfo=timezone.utc),
        {"platform": "example"},
    )
    stamped = stamp_item_properties(stamp_item_time(scene, item), item)
    assert "time" not in stamped.coords
    assert stamped.stac_properties.dims == ("date",)
    observed = scene.gs.write_time_bounds("D")
    assert observed.start_datetime.dims == ("date",)


def test_encoding_time_bounds_leaves_the_callers_raster_unchanged():
    from geosave.geodata.core.raster.utils.cf import (
        decode_time_bounds,
        encode_time_bounds,
    )
    from tests.geodata.conftest import build_dated_raster

    raster = build_dated_raster()

    encoded = encode_time_bounds(raster)
    decoded = decode_time_bounds(encoded)

    assert "bounds" not in raster.time.attrs
    assert "units" not in raster.time.encoding
    assert encoded.time.attrs["bounds"] == "time_bounds"
    assert encoded.time_bounds.dims == ("time", "bounds")
    assert isinstance(encoded.red.data, dask.array.Array)
    for name in ("start_datetime", "end_datetime"):
        np.testing.assert_array_equal(decoded[name].values, raster[name].values)
