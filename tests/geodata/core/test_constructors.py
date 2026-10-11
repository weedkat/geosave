"""Build rasters and arrays on a reference's coordinates."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.constructors import array, raster
from geosave_engine.geodata.io import gdal

from tests.geodata.conftest import build_raster, geobox


def test_raster_dimensions_do_not_depend_on_coordinate_order() -> None:
    pixels = np.zeros((2, 3, 4, 5))
    built = raster(
        {"red": (("time", "depth", "y", "x"), pixels)},
        depth=[1, 2, 3],
        time=[0, 1],
    )

    assert built.red.dims == ("time", "depth", "y", "x")
    assert built.red.data is pixels


def test_raster_explicit_dimensions_allow_unlabelled_and_static_axes() -> None:
    built = raster(
        {
            "red": (("time", "y", "x"), np.zeros((2, 4, 5))),
            "dem": (("y", "x"), np.ones((4, 5))),
        }
    )

    assert built.red.dims == ("time", "y", "x")
    assert built.dem.dims == ("y", "x")
    assert "time" not in built.coords


def test_raster_requires_dimensions_for_bare_arrays() -> None:
    with pytest.raises(TypeError, match="dimensions"):
        raster({"red": np.zeros((4, 5))})


def test_raster_accepts_native_dataarrays_without_computing() -> None:
    import dask.array as da

    band = xr.DataArray(da.ones((4, 5), chunks=(2, 5)), dims=("y", "x"))
    built = raster({"red": band})

    assert built.red.data is band.data


@pytest.mark.parametrize("as_band", [False, True])
def test_native_reference_places_lazy_pixels_on_its_coordinates(
    as_band: bool,
) -> None:
    import dask.array as da

    source = build_raster(times=2, packed=True).assign_coords(
        sun_angle=("time", [20.0, 25.0])
    )
    source.attrs["source_path"] = "imagery.tif"
    reference = source.red if as_band else source
    pixels = da.zeros((2, 2, 2), chunks=(1, 2, 2), dtype="uint8")

    built = raster({"prediction": (("time", "y", "x"), pixels)}, reference)

    assert built.prediction.data is pixels
    assert built.gs.geobox == source.gs.geobox
    assert set(built.coords) == {"time", "sun_angle", "y", "x", "spatial_ref"}
    np.testing.assert_array_equal(built.time, source.time)
    assert not built.attrs
    assert not built.prediction.attrs
    assert built.prediction.encoding == {"grid_mapping": "spatial_ref"}


def test_native_reference_skips_coordinates_of_unused_dimensions() -> None:
    source = build_raster(times=2)

    built = raster({"prediction": (("y", "x"), np.zeros((2, 2)))}, source)

    assert dict(built.sizes) == {"y": 2, "x": 2}
    assert "time" not in built.coords


def test_native_reference_passes_on_scalar_coordinates() -> None:
    source = build_raster(times=2).isel(time=0)

    built = raster({"prediction": (("y", "x"), np.zeros((2, 2)))}, source)

    assert built.time.item() == source.time.item()


def test_native_reference_keeps_exact_sliced_spatial_coordinates() -> None:
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.01, 45.01), crs="EPSG:4326", resolution=1 / 3600
    )
    source = raster({"red": (("y", "x"), np.ones(grid.shape))}, grid).isel(
        x=slice(5, 25), y=slice(5, 25)
    )
    pixels = np.zeros(source.red.shape, dtype="uint8")

    built = raster({"prediction": (("y", "x"), pixels)}, source)

    xr.align(source, built, join="exact")
    assert built.prediction.data is pixels


@pytest.mark.parametrize("as_band", [False, True])
@pytest.mark.parametrize("missing", [("x",), ("y",), ("y", "x")])
def test_native_reference_uses_transform_when_spatial_labels_are_missing(
    missing: tuple[str, ...], as_band: bool
) -> None:
    source = build_raster()
    reference = source.drop_vars(list(missing))
    if as_band:
        reference = reference.red

    built = raster({"prediction": (("y", "x"), np.zeros((2, 2)))}, reference)

    assert built.gs.geobox == source.gs.geobox
    np.testing.assert_array_equal(built.x, source.x)
    np.testing.assert_array_equal(built.y, source.y)


def test_keyword_coordinates_replace_the_references() -> None:
    source = build_raster(times=2)
    dates = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")

    built = raster(
        {"prediction": (("time", "y", "x"), np.zeros((2, 2, 2)))},
        source,
        time=dates,
    )

    np.testing.assert_array_equal(built.time, dates)


def test_native_reference_accepts_bands_derived_from_it() -> None:
    source = build_raster(times=2)

    built = raster({"double": source.red * 2}, source)

    xr.testing.assert_equal(built.double, (source.red * 2).rename("double"))


def test_native_reference_keeps_output_variable_semantics() -> None:
    source = build_raster(packed=True)
    pixels = xr.DataArray(
        np.zeros((2, 2), dtype="float32"),
        dims=("y", "x"),
        attrs={"long_name": "Class confidence", "units": "1"},
    )
    pixels.encoding["scale_factor"] = 0.01

    built = raster({"confidence": pixels}, source, nodata=-1)

    assert built.confidence.attrs["long_name"] == "Class confidence"
    assert built.confidence.attrs["units"] == "1"
    assert built.confidence.gs.nodata == -1
    assert built.confidence.encoding["scale_factor"] == 0.01
    assert built.confidence.data is pixels.data


def test_native_reference_reads_a_custom_crs_coordinate() -> None:
    source = build_raster().rename({"spatial_ref": "crs"})
    for band in source.data_vars.values():
        band.encoding["grid_mapping"] = "crs"

    built = raster({"prediction": (("y", "x"), np.zeros((2, 2)))}, source.red)

    assert built.gs.geobox == source.gs.geobox
    assert built.prediction.encoding["grid_mapping"] == "spatial_ref"
    assert "crs" not in built.coords


@pytest.mark.parametrize(
    ("variable", "coords"),
    [
        ((("y", "x"), np.zeros((1, 2))), {}),
        ((("time", "y", "x"), np.zeros((3, 2, 2))), {}),
        ((("y", "x"), np.zeros((2, 2))), {"x": [0.0, 1.0]}),
    ],
    ids=["spatial shape", "time length", "spatial labels"],
)
def test_native_reference_refuses_pixels_off_its_coordinates(variable, coords) -> None:
    with pytest.raises(ValueError, match="align"):
        raster({"prediction": variable}, build_raster(times=2), **coords)


def test_native_reference_refuses_a_band_on_other_coordinates() -> None:
    source = build_raster()
    shifted = source.red.assign_coords(x=source.x + 1)

    with pytest.raises(ValueError, match="align"):
        raster({"prediction": shifted}, source)


@pytest.mark.parametrize("as_band", [False, True])
def test_native_reference_requires_a_grid(as_band: bool) -> None:
    source = xr.Dataset({"label": (("y", "x"), np.zeros((2, 3)))})
    reference = source.label if as_band else source

    with pytest.raises(ValueError, match="reference has no locatable grid"):
        raster({"prediction": (("y", "x"), np.ones((2, 3)))}, reference)


def test_raster_refuses_a_band_named_like_the_crs_coordinate() -> None:
    source = build_raster()

    with pytest.raises(ValueError, match="spatial_ref"):
        raster({"spatial_ref": (("y", "x"), np.ones((2, 2)))}, source)


def test_raster_refuses_a_tree_as_reference() -> None:
    with pytest.raises(TypeError, match="reference"):
        raster(
            {"prediction": (("y", "x"), np.zeros((2, 2)))},
            xr.DataTree.from_dict({"image": build_raster()}),
        )


@pytest.mark.parametrize("format", ["geotiff", "netcdf"])
def test_native_reference_round_trip(tmp_path: Path, format: str) -> None:
    from geosave_engine.geodata.io import geotiff, netcdf

    source = build_raster(times=2).isel(time=0)
    pixels = np.array([[0, 1], [2, 255]], dtype="uint8")
    written = raster({"prediction": (("y", "x"), pixels)}, source, nodata=255)

    if format == "geotiff":
        destination = geotiff.write_cog(written, tmp_path / "prediction.tif")
        restored = gdal.read(destination)
    else:
        destination = netcdf.write(written, tmp_path / "prediction.nc")
        restored = netcdf.read(destination)

    with restored:
        assert restored.gs.geobox == source.gs.geobox
        np.testing.assert_array_equal(restored.prediction.values.squeeze(), pixels)
        np.testing.assert_array_equal(
            restored.time.values.reshape(-1), source.time.values.reshape(-1)
        )
        assert restored.prediction.dtype == np.dtype("uint8")
        assert restored.prediction.gs.nodata == 255


def test_raster_places_arrays_and_conforms_them() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box)

    assert built.gs.geobox == box
    assert built.gs.variables == ("red",)
    assert built.y.attrs["standard_name"] == "projection_y_coordinate"


def test_raster_names_leading_dims_and_labels_them() -> None:
    box = geobox()
    times = pd.date_range("2024-01-01", periods=2, freq="MS")

    built = raster(
        {
            "red": (
                ("time", *("y", "x")),
                np.zeros((2, *box.shape), "uint16"),
            )
        },
        box,
        time=times,
    )

    assert built.red.dims == ("time", "y", "x")
    assert built.gs.timespan is not None


def test_raster_leaves_an_unlabelled_leading_dim_alone() -> None:
    box = geobox()

    built = raster(
        {
            "probability": (
                ("class", *("y", "x")),
                np.zeros((3, *box.shape), "float32"),
            )
        },
        box,
    )

    assert built.probability.dims == ("class", "y", "x")
    assert "class" not in built.coords


def test_raster_uses_y_x_for_a_geographic_grid() -> None:
    built = raster(
        {
            "red": (
                ("y", "x"),
                np.zeros((4, 4), "uint16"),
            )
        },
        geobox(crs="EPSG:4326", shape=(4, 4)),
    )

    assert built.red.dims == ("y", "x")


def test_raster_without_a_geobox_stays_unplaced() -> None:
    built = raster(
        {
            "band": (
                ("y", "x"),
                np.zeros((4, 4), "uint8"),
            )
        }
    )

    assert built.band.dims == ("y", "x")
    assert not built.coords
    assert built.gs.geobox is None
    assert built.gs.crs_name is None


def test_raster_refuses_no_arrays() -> None:
    with pytest.raises(ValueError, match="at least one named array"):
        raster({})


def test_raster_refuses_arrays_on_different_shapes() -> None:
    with pytest.raises(ValueError, match="conflicting sizes"):
        raster(
            {
                "a": (
                    ("y", "x"),
                    np.zeros((8, 8)),
                ),
                "b": (
                    ("y", "x"),
                    np.zeros((4, 4)),
                ),
            }
        )


def test_raster_refuses_a_rank_that_does_not_match_dims() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="Could not convert tuple"):
        raster({"a": (("y", "x"), np.zeros((3, *box.shape)))}, box)


def test_raster_refuses_trailing_axes_off_the_grid() -> None:
    with pytest.raises(ValueError, match="align"):
        raster({"a": (("y", "x"), np.zeros((4, 4)))}, geobox())


def test_raster_refuses_a_coord_of_the_wrong_length() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="conflicting sizes"):
        raster(
            {
                "a": (
                    ("class", *("y", "x")),
                    np.zeros((3, *box.shape)),
                )
            },
            box,
            **{"class": [1, 2]},
        )


def test_array_builds_the_band_raster_would() -> None:
    from geosave_engine.geodata.core.constructors import array

    pixels = np.ones((2, *geobox().shape), "int16")
    labels = pd.date_range("2024-01-01", periods=2)

    band = array(
        pixels,
        geobox(),
        dims=("time", *("y", "x")),
        time=labels,
        nodata=0,
    )
    variable = raster(
        {
            "a": (
                ("time", *("y", "x")),
                pixels,
            )
        },
        geobox(),
        time=labels,
        nodata=0,
    )["a"]

    assert band.name is None
    xr.testing.assert_identical(band, variable.rename(None))


@pytest.mark.parametrize(
    ("pixels", "dims", "coords", "message"),
    [
        (np.ones((8, 8)), ("time", "y", "x"), {"time": [0]}, "Could not convert tuple"),
        (np.ones((2, 8, 8)), ("time", "y", "x"), {"time": [0]}, "conflicting sizes"),
        (np.ones((4, 4)), ("y", "x"), {}, "align"),
        (np.ones((8, 8)), ("y", "x"), {"x": range(8)}, "align"),
    ],
)
def test_array_refuses_what_raster_refuses(pixels, dims, coords, message) -> None:
    from geosave_engine.geodata.core.constructors import array

    with pytest.raises(ValueError, match=message):
        array(
            pixels,
            geobox(),
            dims=dims,
            **coords,
        )


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857"])
def test_constructor_spatial_names_preserve_cf_grid_through_netcdf(
    tmp_path: Path, crs: str
) -> None:
    import dask.array as da
    from affine import Affine

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), crs)
    pixels = da.ones((2, 3), chunks=(1, 3))
    built = raster({"red": (("y", "x"), pixels)}, grid)

    assert built.red.data is pixels
    assert built.red.dims == ("y", "x")
    assert built.gs.write_crs().red.dims == ("y", "x")
    assert built.y.attrs["axis"] == "Y"
    assert built.x.attrs["axis"] == "X"
    if crs == "EPSG:4326":
        assert built.y.attrs["standard_name"] == "latitude"
        assert built.x.attrs["standard_name"] == "longitude"
        assert built.y.attrs["units"] == "degrees_north"
        assert built.x.attrs["units"] == "degrees_east"
    else:
        assert built.y.attrs["standard_name"] == "projection_y_coordinate"
        assert built.x.attrs["standard_name"] == "projection_x_coordinate"

    path = tmp_path / "grid.nc"
    built.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as restored:
        assert restored.red.dims == ("y", "x")
        assert restored.odc.geobox == grid
        assert restored.rio.crs.to_epsg() == int(crs.split(":")[1])
        assert (restored.rio.y_dim, restored.rio.x_dim) == ("y", "x")
        xr.testing.assert_equal(restored.red, built.red.compute())


def test_a_raster_takes_the_grid_another_raster_reports() -> None:
    source = build_raster()

    built = raster({"copy": (("y", "x"), source.red.values)}, source.gs.geobox)

    assert built.gs.geobox == source.gs.geobox


def test_a_raster_is_built_on_a_grid_placed_by_ground_control_points() -> None:
    from odc.geo.gcp import GCPGeoBox, GCPMapping
    from odc.geo.geom import multipoint

    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 50), (10.1, 50), (10, 49.9), (10.1, 49.9)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))

    built = raster({"red": (("y", "x"), np.zeros((4, 4), "uint8"))}, grid)

    assert isinstance(built.gs.geobox, GCPGeoBox)
    assert built.gs.geobox.shape == grid.shape
    # Its axes count pixels, so no CF standard name may claim they are ground.
    assert "standard_name" not in built.y.attrs
    assert "standard_name" not in built.x.attrs


@pytest.mark.parametrize("as_band", [False, True])
def test_array_takes_native_reference_coordinates(as_band: bool) -> None:
    source = build_raster(times=2)
    reference = source.red if as_band else source
    pixels = da.zeros((2, 2, 2), chunks=(1, 2, 2), dtype="float32")

    built = array(pixels, reference=reference, dims=("time", "y", "x"))

    assert built.name is None
    assert built.data is pixels
    assert built.gs.geobox == source.gs.geobox
    for name in ("time", "y", "x", "spatial_ref"):
        xr.testing.assert_identical(built.coords[name], source.coords[name])
    assert built.encoding["grid_mapping"] == "spatial_ref"


def test_array_takes_a_grid_reference_by_keyword() -> None:
    source = build_raster()
    pixels = np.ones((2, 2), dtype="uint8")

    built = array(pixels, reference=source.gs.geobox, dims=("y", "x"), nodata=255)

    assert built.gs.geobox == source.gs.geobox
    assert built.gs.nodata == 255
    assert built.data is pixels


@pytest.mark.parametrize("as_band", [False, True])
def test_array_takes_a_ground_control_point_reference(as_band: bool) -> None:
    from odc.geo.gcp import GCPGeoBox, GCPMapping
    from odc.geo.geom import multipoint

    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 50), (10.1, 50), (10, 49.9), (10.1, 49.9)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))
    reference = array(np.zeros((4, 4)), grid, dims=("y", "x")) if as_band else grid

    built = array(np.zeros((4, 4), "uint8"), reference=reference, dims=("y", "x"))

    assert isinstance(built.gs.geobox, GCPGeoBox)
    assert built.gs.geobox.shape == grid.shape
    if as_band:
        xr.testing.assert_identical(built.spatial_ref, reference.spatial_ref)
