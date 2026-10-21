import dask.array as da
import pytest
from dask.callbacks import Callback

from geosave.geodata.viz.explorer import RasterLayer


def test_computed_window_executes_only_intersecting_chunks():
    import numpy as np
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    from geosave.geodata.viz.explorer.raster import read_window

    grid = GeoBox.from_bbox(
        (0, 0, 1024, 1024), crs=3857, shape=(1024, 1024), tight=True
    )
    values = da.random.default_rng(0).random((1024, 1024), chunks=(128, 128))
    data = xr.Dataset(
        {"value": xr.DataArray(values * 2, dims=("y", "x"), coords=xr_coords(grid))}
    )
    target = GeoBox.from_bbox((0, 0, 256, 256), crs=3857, shape=(64, 64), tight=True)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(str(key))):
        sampled = read_window(data, target)
    assert sum("random" in key for key in tasks) == 4
    assert sampled.value.shape == (64, 64)
    assert np.isfinite(sampled.value).all()
    assert isinstance(data.value.data, da.Array)


def test_nodata_and_decoded_cog_round_trip(scene, tmp_path):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data["B04"].values[0, 0] = -999
    data = data.gs.write_nodata(-999)
    path = tmp_path / "packed.tif"
    data.gs.to_cog(path)
    loaded = open_raster(path)
    result = read_window(loaded, loaded.odc.geobox)
    assert np.isnan(result.B04.values[0, 0])
    np.testing.assert_allclose(result.B04.values[1:], data.B04.values[1:])


def test_derived_file_values_are_used_and_source_rgb_is_unchanged(scene, tmp_path):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "rgb.tif"
    scene[["B04", "B03", "B02"]].gs.to_cog(path)
    loaded = open_raster(path)
    derived = loaded * 2
    sampled = read_window(derived, derived.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04.compute() * 2)
    layer = RasterLayer(loaded)
    layer.display = "B03"
    assert loaded.gs.rgb == ("B04", "B03", "B02")


def test_changed_values_with_source_encoding_are_authoritative(scene, tmp_path):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "values.tif"
    scene[["B04"]].gs.to_cog(path)
    loaded = open_raster(path)
    changed = loaded.copy()
    changed["B04"] = loaded.B04.copy(data=loaded.B04.data * 3)
    assert "source" in changed.B04.encoding
    sampled = read_window(changed, changed.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04.compute() * 3)


@pytest.mark.parametrize("materialize", ["load", "persist"])
def test_loaded_file_values_do_not_need_the_original_file(scene, tmp_path, materialize):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "values.tif"
    scene[["B04"]].gs.to_cog(path)
    loaded = getattr(open_raster(path), materialize)()
    loaded.close()
    path.unlink()
    sampled = read_window(loaded, loaded.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04)


def test_current_nodata_metadata_overrides_file_nodata(scene, tmp_path):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data.B04.values[0, 0] = -999
    data = data.gs.write_nodata(-999)
    path = tmp_path / "values.tif"
    data.gs.to_cog(path)
    unmasked = open_raster(path).gs.write_nodata(None)
    sampled = read_window(unmasked, unmasked.odc.geobox)
    np.testing.assert_allclose(sampled.B04, unmasked.B04.compute())


def test_decoded_cog_uses_overviews_without_dask_execution(scene, tmp_path):
    import numpy as np

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data.B04.values[0, 0] = -999
    data = data.gs.write_nodata(-999).gs.write_packing(0.5, 2)
    path = tmp_path / "packed.tif"
    data.gs.to_cog(path)
    decoded = open_raster(path, mask_and_scale=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        sampled = read_window(decoded, decoded.odc.geobox)
    np.testing.assert_allclose(sampled.B04.values[1:], data.B04.values[1:] * 0.5 + 2)
    np.testing.assert_allclose(sampled.B04, decoded.B04.compute())
    assert tasks == []


def test_spatial_axis_order_is_normalized_for_display(scene):
    import numpy as np

    from geosave.geodata.viz.explorer.raster import read_window

    transposed = scene[["B04"]].transpose("x", "y")
    sampled = read_window(transposed, transposed.odc.geobox)
    np.testing.assert_allclose(sampled.B04, scene.B04)


def test_nearby_pan_reuses_pixels_but_zoom_reads_detail():
    import numpy as np
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox.from_bbox((0, 0, 4096, 4096), crs=3857, shape=(4096, 4096))
    y, x = da.indices((4096, 4096), chunks=(256, 256))
    data = xr.Dataset(
        {
            "value": xr.DataArray(
                (x + y).astype("float32"), dims=("y", "x"), coords=xr_coords(grid)
            )
        }
    )
    layer = RasterLayer(data)
    layer.render((1024, 2048), (1024, 2048))
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        moved = layer.render((1040, 2064), (1040, 2064))
    assert tasks == [], "a nearby pan must reuse the buffered viewport"
    values = moved.values()[0].dimension_values(2, flat=False)
    assert np.isfinite(values).all()
    assert float(np.nanmax(values)) > float(np.nanmin(values))
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        layer.render((1400, 1500), (1400, 1500))
    assert tasks, "zooming in must request source detail"
    assert isinstance(data.value.data, da.Array)


def test_file_and_computed_bands_keep_their_own_sampling_grids(scene, tmp_path):
    import numpy as np
    from odc.geo.geobox import GeoBox

    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "mixed.tif"
    scene[["B04", "B03"]].gs.to_cog(path)
    loaded = open_raster(path)
    loaded["B03"] = loaded.B03 * 2
    target = GeoBox.from_bbox(loaded.odc.geobox.boundingbox, shape=(23, 29))
    together = read_window(loaded, target)
    for name in loaded.data_vars:
        separate = read_window(loaded[[name]], target)
        np.testing.assert_allclose(together[name], separate[name])
