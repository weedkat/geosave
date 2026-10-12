"""A raster arranged as COG files by a path template."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from geosave.geodata import open_raster
from geosave.geodata.io import cogs

from tests.geodata.conftest import build_raster

DAYS = ("20250601", "20250602")


@pytest.mark.parametrize(
    ("times", "template", "expected"),
    [
        (0, "scene.tif", ["scene.tif"]),
        (0, "s2_{band}.tif", ["s2_red.tif", "s2_nir.tif"]),
        (0, "scene/{band}.tif", ["scene/red.tif", "scene/nir.tif"]),
        (2, "scene_{time:%Y%m%d}.tif", [f"scene_{day}.tif" for day in DAYS]),
        (
            2,
            "{time:%Y%m%d}/{band}.tif",
            [f"{day}/{band}.tif" for day in DAYS for band in ("red", "nir")],
        ),
    ],
    ids=["one-file", "flat-bands", "folder-bands", "dates", "dates-and-bands"],
)
def test_the_template_fields_decide_the_files(
    tmp_path: Path, times: int, template: str, expected: list[str]
) -> None:
    written = build_raster(times=times)

    files = cogs.write(written, f"{tmp_path}/{template}")

    assert [Path(f.href) for f in files] == [tmp_path / name for name in expected]
    restored = open_raster(files)
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        np.testing.assert_array_equal(restored[name].squeeze(), variable.squeeze())


def test_place_is_a_callers_field_not_a_built_in(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="lat"):
        cogs.write(build_raster(), f"{tmp_path}/{{lat:.3f}}.tif")


def test_a_callable_field_names_files_from_the_raster(tmp_path: Path) -> None:
    written = build_raster(times=1)
    lat = written.gs.anchor.lonlat[1]

    (file,) = cogs.write(
        written,
        f"{tmp_path}/{{lat:.3f}}.tif",
        fields={"lat": lambda raster: raster.gs.anchor.lonlat[1]},
    )

    assert Path(file.href) == tmp_path / f"{lat:.3f}.tif"


def test_a_caller_field_is_filled_as_given(tmp_path: Path) -> None:
    (file,) = cogs.write(
        build_raster(), f"{tmp_path}/{{prefix}}.tif", fields={"prefix": "s2"}
    )

    assert Path(file.href) == tmp_path / "s2.tif"


def test_a_callable_field_is_given_each_dates_raster(tmp_path: Path) -> None:
    written = build_raster(times=2)
    seen = []

    def day(raster):
        seen.append(raster)
        return pd.Timestamp(raster.time.values).strftime("%d")

    files = cogs.write(written, f"{tmp_path}/day{{day}}.tif", fields={"day": day})

    assert [Path(f.href) for f in files] == [tmp_path / "day01.tif", tmp_path / "day02.tif"]
    assert all("time" not in raster.dims for raster in seen)


def test_several_dates_without_time_refuse(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match=r"2 dates would share .*scene\.tif; add \{time\}"
    ):
        cogs.write(build_raster(times=2), f"{tmp_path}/scene.tif")
    assert not list(tmp_path.iterdir())


def test_a_partial_date_that_collides_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"make \{time:%Y\} more precise"):
        cogs.write(build_raster(times=2), f"{tmp_path}/{{time:%Y}}.tif")


def test_time_on_an_undated_raster_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no dates, so \{time\} cannot be filled"):
        cogs.write(build_raster(), f"{tmp_path}/{{time:%Y}}.tif")


@pytest.mark.parametrize("field", ["time", "band", "collection"])
def test_a_caller_field_repeating_a_built_in_refuses(tmp_path: Path, field) -> None:
    with pytest.raises(TypeError, match=f"{field}= repeats the field"):
        cogs.write(build_raster(), f"{tmp_path}/x.tif", fields={field: "a"})


def test_a_path_that_is_not_a_tiff_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end the path in .tif"):
        cogs.write(build_raster(), f"{tmp_path}/scene")


def test_sub_second_instants_render_through_strftime(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])

    (file,) = cogs.write(written, f"{tmp_path}/{{time:%Y%m%dT%H%M%S_%f}}.tif")

    assert Path(file.href).name == "20250601T103031_123456.tif"
    assert open_raster(file.href).time.values[0] == instant


def test_joined_bands_keep_the_variable_order(tmp_path: Path) -> None:
    written = build_raster(times=2)

    paths = cogs.write(written, f"{tmp_path}/{{time:%Y%m%d}}.tif")

    assert list(open_raster(paths).data_vars) == list(written.data_vars)


def test_map_scale_reaches_every_file(tmp_path: Path) -> None:
    source = build_raster(times=2)
    written = source.assign(
        {name: source[name].assign_attrs(units="reflectance") for name in ("red", "nir")}
    )

    files = cogs.write(written, f"{tmp_path}/{{time:%Y%m%d}}.tif", map_scale=10_000)

    for file in files:
        with rasterio.open(file.href) as src:
            assert float(src.tags()["TIFFTAG_XRESOLUTION"]) == pytest.approx(10.0)
    assert open_raster(files).red.attrs["units"] == "reflectance"


def test_an_existing_file_refuses_without_overwrite(tmp_path: Path) -> None:
    written = build_raster(times=1)
    cogs.write(written, f"{tmp_path}/scene.tif")

    with pytest.raises(FileExistsError):
        cogs.write(written, f"{tmp_path}/scene.tif")
    assert cogs.write(written, f"{tmp_path}/scene.tif", overwrite=True)


def test_a_raster_spanning_more_than_time_refuses(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a COG holds one instant of one grid"):
        cogs.write(spanning, f"{tmp_path}/{{time:%Y%m%d}}.tif")


def test_remote_paths_keep_their_scheme(bucket: str) -> None:
    files = cogs.write(build_raster(times=2), f"{bucket}/{{time:%Y%m%d}}/{{band}}.tif")

    assert [f.href for f in files] == [
        f"{bucket}/{day}/{band}.tif" for day in DAYS for band in ("red", "nir")
    ]


def test_layout_gives_each_file_its_date_and_bands() -> None:
    files = cogs.layout(build_raster(times=2), "out/{time:%Y%m%d}.tif")

    assert [file.path for file in files] == [f"out/{day}.tif" for day in DAYS]
    assert files[0].time == pd.Timestamp("2025-06-01")
    assert files[0].raster.gs.variables == ("red", "nir")
    assert "time" not in files[0].raster.dims


def test_write_writes_exactly_the_layout(tmp_path: Path) -> None:
    source = build_raster(times=2)
    template = f"{tmp_path}/{{time:%Y%m%d}}/{{band}}.tif"

    written = cogs.write(source, template)

    assert [f.href for f in written] == [
        file.path for file in cogs.layout(source, template)
    ]


def test_to_cog_returns_one_record_per_file(tmp_path: Path) -> None:
    from geosave.geodata.io import CogFile

    scene = build_raster(times=2)

    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif")

    assert all(isinstance(file, CogFile) for file in files)
    assert [f.bands for f in files] == [("red",), ("nir",), ("red",), ("nir",)]
    assert len({f.href for f in files}) == 4
    assert files[0].datetime == pd.Timestamp(scene.time.values[0])
    assert files[0].collection is None


def test_an_unsplit_file_lists_every_band_in_order(tmp_path: Path) -> None:
    (file,) = build_raster().gs.to_cog(f"{tmp_path}/scene.tif")

    assert file.bands == ("red", "nir")
    assert file.datetime is None
    assert Path(file.href).is_absolute()


def test_a_stack_names_each_file_by_its_collection(tmp_path: Path) -> None:
    from geosave.geodata import stack

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    sample = stack({"optical": image, "label": label})

    files = sample.gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    assert [f.collection for f in files] == ["optical", "label"]
    assert Path(files[0].href).name.startswith("optical_")
    assert Path(files[1].href).name == "label.tif"


def test_records_read_back_as_the_raster_written(tmp_path: Path) -> None:
    scene = build_raster(times=2)
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif")

    restored = cogs.read(files)

    assert list(restored.data_vars) == ["red", "nir"]
    assert restored.sizes["time"] == 2
    np.testing.assert_array_equal(restored.red.values, scene.red.values)


def test_a_stack_reads_back_from_its_records(tmp_path: Path) -> None:
    from geosave.geodata import open_stack, stack

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    files = stack({"optical": image, "label": label}).gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    restored = open_stack(files)

    assert restored.gs.groups == ("optical", "label")
    assert "time" not in restored["label"].dims


def test_records_read_onto_a_grid_stay_lazy(tmp_path: Path) -> None:
    import dask

    scene = build_raster(times=2)
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    grid = scene.gs.geobox

    read = cogs.read(files, geobox=grid)

    assert dask.is_dask_collection(read.red)
    assert read.gs.geobox == grid


def test_a_grid_inside_the_files_own_is_sliced_not_warped(tmp_path: Path) -> None:
    import warnings

    from geosave.geodata.warnings import AssumedFillWarning

    scene = build_raster(times=2)
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    window = scene.gs.geobox[1:2, 0:2]

    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumedFillWarning)
        read = cogs.read(files, geobox=window)

    assert read.gs.geobox == window
    np.testing.assert_array_equal(read.red.values, scene.red.values[:, 1:2, 0:2])


def test_a_file_placed_by_control_points_is_warped_not_sliced(tmp_path: Path) -> None:
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS

    from geosave.geodata.io import gdal

    source = tmp_path / "gcp.tif"
    corners = [
        GroundControlPoint(row=0, col=0, x=3e5, y=5e6),
        GroundControlPoint(row=8, col=8, x=3e5 + 80, y=5e6 - 80),
        GroundControlPoint(row=0, col=8, x=3e5 + 80, y=5e6),
    ]
    with rasterio.open(
        source, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8"
    ) as dst:
        dst.write(np.ones((1, 8, 8), "uint8"))
        dst.gcps = (corners, CRS.from_epsg(32633))
    placed = gdal.read(source).gs.geobox

    # A control-point grid has no affine transform to slice by.
    assert cogs._window(placed, build_raster().gs.geobox) is None


def test_a_scalar_time_fills_the_time_field(tmp_path: Path) -> None:
    # A scalar time coordinate is one date, as one step of a time axis is.
    scene = build_raster(times=2).isel(time=1)

    files = cogs.write(scene, f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    assert [Path(file.href).name for file in files] == ["s_20250602.tif"]
    assert files[0].datetime == pd.Timestamp("2025-06-02")
    restored = open_raster(files[0].href)
    assert pd.DatetimeIndex(restored.time.values).tolist() == [
        pd.Timestamp("2025-06-02")
    ]


def _unnamed_tiff(
    path: Path, resolution: float, value: int, nodata: int | None = 0
) -> str:
    """Write a one-band GeoTIFF with no band description, as STAC hosts serve them."""
    from rasterio.transform import from_origin

    size = round(40 / resolution)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint16",
        crs="EPSG:32749",
        transform=from_origin(500_000.0, 9_000_040.0, resolution, resolution),
        nodata=nodata,
    ) as tiff:
        tiff.write(np.full((1, size, size), value, dtype="uint16"))
    return str(path)


def test_records_name_the_bands_a_file_leaves_unnamed(tmp_path: Path) -> None:
    files = [
        cogs.CogFile(_unnamed_tiff(tmp_path / f"{band}.tif", 10, value), (band,), None)
        for band, value in (("B02", 2), ("B03", 3))
    ]

    read = cogs.read(files)

    assert list(read.data_vars) == ["B02", "B03"]
    assert int(read.B03.max()) == 3


def test_bands_of_mixed_resolution_read_onto_one_grid(tmp_path: Path) -> None:
    from odc.geo.geobox import GeoBox

    files = [
        cogs.CogFile(_unnamed_tiff(tmp_path / "B02.tif", 10, 2), ("B02",), None),
        cogs.CogFile(_unnamed_tiff(tmp_path / "SCL.tif", 20, 4), ("SCL",), None),
    ]
    window = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        resolution=10,
    )

    read = cogs.read(files, geobox=window)

    assert list(read.data_vars) == ["B02", "SCL"]
    assert read.gs.geobox == window
    assert int(read.SCL.max()) == 4


def test_bands_without_fill_warn_once_per_read(tmp_path: Path) -> None:
    import warnings

    from affine import Affine
    from odc.geo.geobox import GeoBox

    from geosave.geodata.warnings import AssumedFillWarning

    files = [
        cogs.CogFile(
            _unnamed_tiff(tmp_path / f"{band}.tif", 10, 2, nodata=None), (band,), None
        )
        for band in ("B02", "B03")
    ]
    # Half a pixel off the files' own grid, so every band is warped.
    shifted = GeoBox(
        (3, 3), Affine(10, 0, 500_005.0, 0, -10, 9_000_035.0), "EPSG:32749"
    )

    messages = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cogs.read(files, geobox=shifted)
        messages.append(
            [
                str(w.message)
                for w in caught
                if issubclass(w.category, AssumedFillWarning)
            ]
        )

    first, second = messages
    assert len(first) == 1
    assert "B02" in first[0] and "B03" in first[0]
    assert first == second


def test_one_name_over_several_bands_numbers_them(tmp_path: Path) -> None:
    from rasterio.transform import from_origin

    path = tmp_path / "visual.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=2, height=2, count=3, dtype="uint8",
        crs="EPSG:32749", transform=from_origin(500_000.0, 9_000_020.0, 10, 10),
    ) as tiff:
        tiff.write(np.stack([np.full((2, 2), value, "uint8") for value in (1, 2, 3)]))

    read = cogs.read([cogs.CogFile(str(path), ("visual",), None)])

    assert list(read.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert int(read["visual.3"].max()) == 3
