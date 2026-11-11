from datetime import datetime as dt

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dask.callbacks import Callback

from geosave.geodata import GeoDataset
from geosave.geodata.core.raster.chips import Chips
from tests.geodata.conftest import build_raster, flat
from tests.geodata.conftest import create_test_collection


def test_to_catalog_writes_current_pixels_with_collection_time_and_band(tmp_path):
    import geopandas as gpd
    import geosave as gs

    original = build_raster(times=2)
    changed = original.copy()
    changed["red"] = changed.red + 1
    collection = create_test_collection("s2", description="Reflectance")
    rows = changed.gs.to_catalog(
        tmp_path,
        format_paths="{collection}/{time:%Y%m%d}/{variable}.tif",
        collection=collection,
    )
    assert isinstance(rows, gpd.GeoDataFrame)
    assert rows.collection.tolist() == ["s2", "s2"]
    assert rows.gs.collections["s2"].description == "Reflectance"
    assert (tmp_path / "s2/20250601/red.tif").is_file()
    assert (tmp_path / "s2/20250602/nir.tif").is_file()
    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")
    restored = gs.open_vector(saved).gs.to_datatree()
    raster = restored["s2"].gs.to_dataset()
    assert raster.red.chunks is not None
    np.testing.assert_array_equal(raster.red.values, changed.red.values)
    np.testing.assert_array_equal(raster.nir.values, changed.nir.values)
    assert raster.gs.geobox == changed.gs.geobox
    restored.close()


def test_to_catalog_refuses_unsplit_dates_before_writing(tmp_path):
    with pytest.raises(ValueError, match="GeoTIFF"):
        build_raster(times=2).gs.to_catalog(
            tmp_path, format_paths="scene.tif", split_item=(), split_asset=()
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_to_catalog_refuses_store_templates_before_writing(tmp_path, suffix):
    with pytest.raises(ValueError, match="GeoTIFF"):
        build_raster(times=2).gs.to_catalog(
            tmp_path, format_paths="{time:%Y%m%d}" + suffix, split_asset=()
        )
    assert not list(tmp_path.iterdir())


def test_to_catalog_dataarray_keeps_its_band_name_and_refuses_overwrite(tmp_path):
    source = build_raster(times=1).red
    rows = source.gs.to_catalog(tmp_path, format_paths="{variable}.tif")
    assert set(rows.iloc[0].assets) == {"red"}
    with pytest.raises(FileExistsError):
        source.gs.to_catalog(tmp_path, format_paths="{variable}.tif")


def test_to_catalog_refuses_unlocated_rasters_before_writing(tmp_path):
    source = xr.Dataset({"red": (("y", "x"), np.ones((2, 2), dtype="float32"))})
    with pytest.raises(ValueError, match="grid"):
        source.gs.to_catalog(tmp_path, format_paths="scene.tif")
    assert not list(tmp_path.iterdir())


def test_raster_accessor_survives_xarray_selection(raster: xr.Dataset) -> None:
    selected = raster.isel(x=slice(0, 1))

    assert isinstance(selected.gs, GeoDataset)
    assert selected.gs.variables == ("red", "nir")
    geobox = selected.gs.geobox
    assert geobox is not None
    assert geobox.width == 1


def test_times_format_as_a_datetime_index_on_a_raster_and_a_band() -> None:
    raster = build_raster(times=2)

    assert isinstance(raster.gs.times, pd.DatetimeIndex)
    assert raster.gs.times.strftime("%Y%m%d").tolist() == ["20250601", "20250602"]
    assert raster.red.gs.times.equals(raster.gs.times)


@pytest.mark.parametrize(
    ("labels", "span"),
    [
        (["2018-12-26"], ("2018-12-26T00:00:00", "2018-12-26T00:00:00")),
        (["2018-12-26T10:30:31"], ("2018-12-26T10:30:31", "2018-12-26T10:30:31")),
        (
            ["2018-12-27", "2018-12-26"],
            ("2018-12-26T00:00:00", "2018-12-27T00:00:00"),
        ),
    ],
    ids=["an instant", "an instant with seconds", "oldest to newest"],
)
def test_instants_span_from_the_oldest_to_the_newest_label(
    labels: list[str], span: tuple[str, str]
) -> None:
    raster = build_raster(times=len(labels)).assign_coords(
        time=np.array(labels, dtype="datetime64[ns]")
    )

    assert raster.gs.timespan == tuple(dt.fromisoformat(edge) for edge in span)


def _foreign() -> xr.Dataset:
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    built = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape))},
        coords=xr_coords(grid),
    )
    built.red.encoding["grid_mapping"] = "spatial_ref"
    return built


def test_gs_reads_native_spatial_names() -> None:
    foreign = _foreign()
    tree = xr.DataTree.from_dict({"/optical": foreign})

    for held in (foreign, foreign.red, tree):
        held.gs.validate(georeferenced=True)
        assert held.gs.geobox == foreign.odc.geobox


def test_gs_reads_the_grid_once_it_is_renamed() -> None:
    foreign = _foreign()

    renamed = foreign.rename({"latitude": "y", "longitude": "x"})

    assert renamed.gs.geobox == foreign.odc.geobox
    assert renamed.gs.write_crs().y.attrs["standard_name"] == "latitude"


@pytest.mark.parametrize("kind", ["raster", "array", "tree"])
@pytest.mark.parametrize("lazy", [False, True])
def test_mask_and_scale_decodes_values_and_keeps_the_source(
    kind: str, lazy: bool
) -> None:
    scene = build_raster(packed=True)
    scene.attrs["title"] = "Stored scene"
    scene.red.attrs.update(long_name="Red", units="1", add_offset=np.float32(0.5))
    if lazy:
        scene = scene.chunk({"y": 1, "x": 1})
    if kind == "array":
        source = scene.red
    elif kind == "tree":
        source = xr.DataTree.from_dict(
            {"/": xr.Dataset(attrs={"title": "Stack"}), "/optical": scene}
        )
    else:
        source = scene
    before = source.copy(deep=True)

    decoded = source.gs.mask_and_scale()
    band = (
        decoded["optical/red"]
        if kind == "tree"
        else decoded
        if kind == "array"
        else decoded.red
    )

    assert isinstance(decoded, type(source))
    assert isinstance(band.data, da.Array) == lazy
    np.testing.assert_allclose(band.compute().values, [[0.6, 0.7], [0.8, np.nan]])
    assert "scale_factor" not in band.attrs
    assert "add_offset" not in band.attrs
    assert "_FillValue" not in band.attrs
    assert band.attrs["long_name"] == "Red"
    assert band.attrs["units"] == "1"
    assert band.gs.geobox == scene.gs.geobox
    if kind != "array":
        assert decoded.attrs == source.attrs
    xr.testing.assert_identical(decoded.gs.mask_and_scale(), decoded)
    xr.testing.assert_identical(source, before)


def test_a_band_read_from_disk_uses_native_cf_precision(tmp_path) -> None:
    from geosave.geodata import open_raster

    (written,) = flat(
        build_raster(packed=True).gs.to_cog(
            tmp_path, format_paths="scene.tif", split_asset=()
        )
    )

    with open_raster(written) as raw:
        decoded = raw.gs.mask_and_scale()
        expected = xr.decode_cf(raw)
        assert decoded.red.dtype == expected.red.dtype
        assert decoded.red.encoding["dtype"] == raw.red.dtype
        xr.testing.assert_allclose(decoded.red, expected.red)


def _dated(labels, *, dims=("time",), as_dataset=False):
    """Lazy raster whose time labels sit on the given dimension."""
    times = np.array(labels, dtype="datetime64[ns]")
    raster = xr.DataArray(
        da.ones((times.size, 2), chunks=(1, 2)),
        dims=(*dims, "x"),
        coords={"time": (dims, times)},
    )
    return raster.to_dataset(name="image") if as_dataset else raster


@pytest.mark.parametrize("as_dataset", [False, True])
def test_monthly_labels_state_month_cells_without_reading_pixels(as_dataset):
    raster = _dated(["2025-01-01", "2025-02-01"], as_dataset=as_dataset)

    dated = raster.gs.write_time_bounds("MS")

    np.testing.assert_array_equal(
        dated.start_datetime.values,
        np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        dated.end_datetime.values,
        np.array(["2025-02-01", "2025-03-01"], dtype="datetime64[ns]"),
    )
    original = raster.image if as_dataset else raster
    result = dated.image if as_dataset else dated
    assert result.data is original.data
    assert "start_datetime" not in raster.coords


def test_right_labels_state_the_cell_before_them():
    dated = _dated(["2025-01-05", "2025-01-12"]).gs.write_time_bounds(
        "W", label="right"
    )

    np.testing.assert_array_equal(
        dated.start_datetime.values,
        np.array(["2024-12-29", "2025-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(dated.end_datetime.values, dated.time.values)


def test_cells_follow_an_auxiliary_time_dimension():
    raster = _dated(["2025-01-01", "2025-02-01"], dims=("observation",))

    dated = raster.gs.write_time_bounds("MS")

    assert dated.start_datetime.dims == ("observation",)
    assert dated.end_datetime.dims == ("observation",)
    assert dated.sizes == raster.sizes


def test_scalar_time_states_scalar_cells():
    dated = _dated(["2025-01-01"]).isel(time=0).gs.write_time_bounds("MS")

    assert dated.start_datetime.ndim == 0
    assert dated.end_datetime.values == np.datetime64("2025-02-01", "ns")


def test_labels_off_the_frequency_are_refused():
    with pytest.raises(ValueError, match="not on a MS edge"):
        _dated(["2025-01-10"]).gs.write_time_bounds("MS")


def test_cells_need_a_time_coordinate():
    with pytest.raises(KeyError, match="temporal coordinate"):
        xr.DataArray([1.0], dims="x").gs.write_time_bounds("MS")


@pytest.mark.parametrize("as_dataset", [False, True])
def test_timespan_follows_selected_cells_and_empty_time(as_dataset):
    from datetime import datetime

    raster = _dated(["2025-01-01", "2025-02-01"], as_dataset=as_dataset)
    dated = raster.gs.write_time_bounds("MS")

    assert dated.isel(time=1).gs.timespan == (
        datetime(2025, 2, 1),
        datetime(2025, 3, 1),
    )
    assert dated.isel(time=1).drop_vars("time").gs.timespan == (
        datetime(2025, 2, 1),
        datetime(2025, 3, 1),
    )
    assert dated.isel(time=slice(0, 0)).gs.timespan is None
    assert raster.isel(time=slice(0, 0)).gs.timespan is None


def test_selective_validation_distinguishes_pixels_grid_and_crs():
    image = xr.DataArray(da.zeros((3, 4)), dims=("y", "x"), name="image")
    image.gs.validate()
    with pytest.raises(ValueError, match="image.*grid"):
        image.gs.validate(georeferenced=True)
    located = image.assign_coords(y=[2.0, 1.0, 0.0], x=[0.0, 1.0, 2.0, 3.0])
    located.gs.validate()
    assert not located.gs.is_georeferenced
    with pytest.raises(ValueError, match="image.*CRS"):
        located.gs.validate(georeferenced=True)
    mixed = xr.Dataset({"image": image, "label": (("row", "col"), np.ones((3, 4)))})
    with pytest.raises(ValueError, match="label.*spatial"):
        mixed.gs.validate()


def test_tree_validation_allows_different_grids_unless_requested():
    source = build_raster()
    tree = xr.DataTree.from_dict(
        {"native": source, "crop": source.isel(x=slice(1, None))}
    )
    tree.gs.validate(georeferenced=True)
    with pytest.raises(ValueError, match="crop.*grid"):
        tree.gs.validate(same_grid=True)


def test_native_axes_support_inspection_validation_and_chips():
    source = build_raster().rename(y="latitude", x="longitude")
    source.gs.validate(georeferenced=True, same_grid=True)
    assert source.red.gs.geobox == source.odc.geobox
    chips = Chips([source], size=3, constant_value=0)
    predictions = []
    for key in chips:
        chip = chips[key]
        assert chip.gs.spatial_dims == ("latitude", "longitude")
        predictions.append((key, np.ones(chip.gs.geobox.shape)))
    pixels = chips.merge(predictions)[0]
    reference = chips.xarray[0]
    merged = xr.DataArray(
        pixels, dims=reference.gs.spatial_dims, coords=reference.coords
    )
    merged.encoding["grid_mapping"] = str(reference.odc.crs_coord.name)
    assert merged.dims == ("latitude", "longitude")
    assert merged.gs.geobox == source.gs.geobox


def test_gcp_inventory_does_not_claim_an_affine_transform():
    from odc.geo.gcp import GCPGeoBox, GCPMapping
    from odc.geo.geom import multipoint

    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 20), (14, 20), (10, 16), (14, 16)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))
    image = xr.DataArray(np.ones((4, 4)), dims=("y", "x")).gs.write_grid(grid)
    image.gs.validate(georeferenced=True)
    with pytest.raises(ValueError, match="ground control points"):
        image.gs.write_crs()
    with pytest.raises(ValueError, match="ground control points"):
        image.to_dataset(name="image").gs.read_pixels(build_raster().gs.geobox)
    with pytest.raises(ValueError, match="ground control points"):
        Chips([image.to_dataset(name="image")], size=4)
    assert isinstance(image.gs.geobox, GCPGeoBox)


def test_cf_axes_can_be_inspected_without_renaming():
    source = build_raster().rename(y="north", x="east")
    source.north.attrs["axis"] = "Y"
    source.east.attrs["axis"] = "X"
    source.gs.validate(georeferenced=True, same_grid=True)
    assert source.red.dims == ("north", "east")
    assert source.gs.geobox == build_raster().gs.geobox
    with pytest.raises(ValueError, match="spatial"):
        source.gs.expand_rasters("north")


def test_empty_and_scalar_inputs_are_inspectable_but_fail_spatial_validation():
    assert xr.Dataset().gs.geobox is None
    scalar = xr.DataArray(1, name="version")
    assert scalar.gs.geobox is None
    for value in (
        xr.Dataset(),
        scalar,
        xr.DataArray(np.zeros((0, 2)), dims=("y", "x")),
    ):
        with pytest.raises(ValueError):
            value.gs.validate()


def test_native_axes_keep_numpy_layout_and_crs_writes():
    source = build_raster().rename(y="latitude", x="longitude")
    np.testing.assert_array_equal(source.gs.to_numpy(), build_raster().gs.to_numpy())
    np.testing.assert_array_equal(source.red.gs.to_numpy(), source.red.values)
    assert source.red.gs.axes == {}
    referenced = source.gs.write_crs()
    assert (
        referenced.latitude.attrs["standard_name"] == "latitude"
        or referenced.latitude.attrs["standard_name"] == "projection_y_coordinate"
    )
    assert referenced.gs.geobox == source.gs.geobox


@pytest.mark.parametrize("x", [[0.0, 1.0, 5.0], [0.0, 0.0, 0.0], [0.0, np.nan, 2.0]])
def test_grid_checks_reject_irregular_degenerate_or_nonfinite_axes(x):
    image = xr.DataArray(
        da.ones((3, 3)),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0, 0.0], "x": x},
        name="image",
    )
    with pytest.raises(ValueError, match="image.*(grid|regular|finite|repeats)"):
        image.gs.validate()


def test_shared_grid_requires_placement_but_pixel_layout_does_not():
    image = xr.DataArray(da.zeros((3, 4)), dims=("y", "x"), name="image")
    pixels = xr.Dataset({"image": image, "label": image.astype("uint8")})
    pixels.gs.validate(same_dims=True)
    with pytest.raises(ValueError, match="grid"):
        pixels.gs.validate(same_grid=True)
    mixed = xr.Dataset(
        {"image": image, "label": (("latitude", "longitude"), np.zeros((3, 4)))}
    )
    mixed.gs.validate()
    with pytest.raises(ValueError, match="label.*spatial dimensions"):
        mixed.gs.validate(same_dims=True)
    located = image.assign_coords(y=[2.0, 1.0, 0.0], x=[0.0, 1.0, 2.0, 3.0])
    tree = xr.DataTree.from_dict({"pixels": pixels, "located": located.to_dataset()})
    with pytest.raises(ValueError, match="pixels/image.*grid") as caught:
        tree.gs.validate(same_grid=True)
    assert "write_grid" in str(caught.value)
    mixed_placement = xr.Dataset(
        {"image": image, "located": located.rename(y="latitude", x="longitude")}
    )
    with pytest.raises(ValueError, match="write_grid"):
        mixed_placement.gs.validate(same_grid=True)
    with pytest.raises(ValueError, match="grid"):
        pixels.gs.validate(georeferenced=True)


def test_coverage_and_crs_annotation_reject_irregular_grids():
    image = xr.DataArray(
        np.ones((3, 3)),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 5.0]},
        name="image",
    )
    image = image.odc.assign_crs("EPSG:4326")
    with pytest.raises(ValueError, match="regularly spaced"):
        _ = image.gs.region
    with pytest.raises(ValueError, match="regularly spaced"):
        image.gs.write_crs()


def test_spatial_dims_describes_native_axes_and_tree_agreement_without_pixels():
    source = build_raster().chunk().rename(y="latitude", x="longitude")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        assert source.gs.spatial_dims == ("latitude", "longitude")
        assert source.red.gs.spatial_dims == ("latitude", "longitude")
        tree = xr.DataTree.from_dict(
            {
                "metadata": xr.Dataset(attrs={"title": "scene"}),
                "image": source,
                "crop": source.isel(longitude=slice(1, None)),
            }
        )
        assert tree.gs.spatial_dims == ("latitude", "longitude")
        mixed = xr.DataTree.from_dict({"image": source, "native": build_raster()})
        assert mixed.gs.spatial_dims is None
        assert xr.DataTree().gs.spatial_dims is None
        assert xr.DataArray(1).gs.spatial_dims is None
    assert tasks == []


def test_spatial_dims_reads_cf_axes_without_assuming_dimension_order():
    source = build_raster().rename(y="north", x="east").expand_dims(level=[1], axis=-1)
    source.north.attrs["axis"] = "Y"
    source.east.attrs["standard_name"] = "projection_x_coordinate"
    assert source.gs.spatial_dims == ("north", "east")
    assert source.red.gs.spatial_dims == ("north", "east")


def test_dataset_grid_inspection_requires_agreement_of_every_variable():
    source = build_raster().chunk()
    mixed_axes = xr.Dataset(
        {
            "red": source.red,
            "native": source.nir.rename(y="latitude", x="longitude"),
        }
    )
    scalar = source.assign(version=xr.DataArray(1))
    cropped = xr.Dataset(
        {
            "red": source.red,
            "crop": source.nir.isel(x=slice(1, None)).rename(
                y="latitude", x="longitude"
            ),
        }
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        assert mixed_axes.gs.spatial_dims is None
        assert mixed_axes.gs.geobox == source.gs.geobox
        mixed_axes.gs.validate(same_grid=True)
        assert scalar.gs.spatial_dims is None
        assert scalar.gs.geobox is None
        assert cropped.gs.geobox is None
        with pytest.raises(ValueError, match="crop.*grid"):
            cropped.gs.validate(same_grid=True)
        tree = xr.DataTree.from_dict({"mixed": scalar})
        assert tree.gs.spatial_dims is None
        assert tree.gs.geobox is None
    assert tasks == []


@pytest.mark.parametrize("value", [xr.Dataset(), xr.DataTree(), xr.DataArray(1)])
def test_basic_validation_rejects_empty_and_nonspatial_inputs(value):
    with pytest.raises(ValueError):
        value.gs.validate()


def test_validation_errors_explain_correction_and_preserve_tree_context():
    from geosave.geodata import RasterValidationError

    image = xr.DataArray(
        da.ones((2, 2)),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="red",
    )
    tree = xr.DataTree.from_dict({"observations/optical": image.to_dataset()})
    with pytest.raises(RasterValidationError) as caught:
        tree.gs.validate(georeferenced=True, same_grid=True)
    error = caught.value
    assert error.path == "observations/optical"
    assert error.variable == "red"
    assert "write_crs" in str(error)
    assert "georeferenced=True" not in str(error)
    assert isinstance(error, ValueError)

    with pytest.raises(RasterValidationError) as caught:
        image.to_dataset().gs.read_pixels(build_raster().gs.geobox)
    assert caught.value.operation == "Cannot read the requested window"
    assert "write_crs" in str(caught.value)


def test_validation_observes_metadata_changes_on_the_same_array():
    from geosave.geodata import RasterValidationError

    image = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0]},
        name="red",
    )
    image.gs.validate()
    image.coords["x"] = [0.0, 1.0, 4.0]
    with pytest.raises(RasterValidationError, match="regularly spaced"):
        image.gs.validate()


def test_cells_follow_a_named_label_coordinate():
    raster = _dated(["2025-01-01", "2025-02-01"]).rename(time="acquired")

    dated = raster.gs.write_time_bounds("MS", dim="acquired")

    assert dated.start_datetime.dims == ("acquired",)
