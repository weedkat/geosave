from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from upath import UPath

from geosave.geodata import attrs
from geosave.geodata.io import gdal, geotiff
from geosave.geodata.warnings import UnreadMaskWarning
from tests.geodata.conftest import build_raster


def _interpreted(ds: xr.Dataset, **colours: str) -> xr.Dataset:
    """Return the raster with the named variables drawn as the given colours."""
    return ds.assign(
        {
            name: attrs.stamp_attrs(ds[name], attrs.GDALBand(colorinterp=colour))
            for name, colour in colours.items()
        }
    )


def _colorinterp(band: xr.DataArray) -> str | None:
    found = attrs.GDALBand.from_attrs(band.attrs)
    return None if found is None else found.colorinterp


def test_round_trip_keeps_the_variable_names(tmp_path: Path) -> None:
    written = build_raster(times=0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert list(restored.data_vars) == list(written.data_vars)
    assert restored.odc.geobox == written.odc.geobox
    for name, variable in written.data_vars.items():
        assert restored[name].dtype == np.dtype("uint16")
        assert np.array_equal(restored[name].values, variable.values)


def test_identity_and_the_cf_label_take_separate_slots(tmp_path: Path) -> None:
    written = build_raster(times=0)
    first = next(iter(written.data_vars))
    written[first].attrs["long_name"] = "Red reflectance (665 nm)"

    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        # The description is CF's label; identity rides in the band's own metadata.
        assert src.descriptions[0] == "Red reflectance (665 nm)"
        assert src.tags(1)["variable_name"] == first


def test_a_cf_long_name_survives_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0)
    first = next(iter(written.data_vars))
    written[first].attrs["long_name"] = "Red reflectance (665 nm)"

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored[first].attrs["long_name"] == "Red reflectance (665 nm)"


def test_a_band_described_by_its_name_gains_no_long_name(tmp_path: Path) -> None:
    # rioxarray describes an unlabelled band by its variable name.
    written = build_raster(times=0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    for name in written.data_vars:
        assert "long_name" not in restored[name].attrs


def test_storage_options_are_accepted_and_left_to_gdal(tmp_path: Path) -> None:
    path = geotiff.write_cog(build_raster(times=0), tmp_path / "scene.tif")

    restored = gdal.open_raster(path, storage_options={"anon": True})

    assert list(restored.data_vars) == ["red", "nir"]


def test_a_scalar_time_survives_as_a_datetime_tag(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)
    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    restored = gdal.open_raster(path)

    with rasterio.open(path) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"
    assert restored.time.values == instant
    # The date lives on the time coordinate alone, so dates join without conflict.
    assert "TIFFTAG_DATETIME" not in restored.attrs
    assert "GEOSAVE_DATETIME" not in restored.attrs


@pytest.mark.parametrize("dated", ["tag", "name"])
def test_a_dated_file_reads_its_date_as_a_scalar_time(
    tmp_path: Path, dated: str
) -> None:
    written = build_raster(times=0)
    if dated == "tag":
        written = written.assign_coords(time=np.datetime64("2024-03-05"))
    name = "scene.tif" if dated == "tag" else "scene_20240305.tif"

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / name))

    # A file holds one instant, so its bands keep only their spatial axes.
    assert "time" not in restored.dims
    assert restored.time.values == np.datetime64("2024-03-05")


@pytest.mark.parametrize("dated", [False, True])
def test_arithmetic_on_an_opened_file_stays_lazy(tmp_path: Path, dated: bool) -> None:
    name = "scene_20240305.tif" if dated else "scene.tif"
    path = geotiff.write_cog(build_raster(times=0), tmp_path / name)

    scene = gdal.open_raster(path)
    scaled = scene["red"] * 2

    assert isinstance(scaled.data, da.Array)


@pytest.mark.parametrize("dated", [False, True])
def test_opening_a_file_reads_no_pixels(tmp_path: Path, dated: bool) -> None:
    name = "scene_20240305.tif" if dated else "scene.tif"
    path = geotiff.write_cog(build_raster(times=0), tmp_path / name)

    restored = gdal.open_raster(path)

    assert not any(restored[name].variable._in_memory for name in restored.data_vars)


def test_an_undated_file_reads_with_no_time(tmp_path: Path) -> None:
    restored = gdal.open_raster(
        geotiff.write_cog(build_raster(times=0), tmp_path / "a.tif")
    )

    assert "time" not in restored.coords


@pytest.mark.parametrize(
    ("stem", "instant"),
    [
        ("20190507_red", "2019-05-07T00:00:00"),
        ("20250601T000000_red", "2025-06-01T00:00:00"),
        ("20190507-20190509_red", "2019-05-07T00:00:00"),
    ],
    ids=["a day", "an instant", "a period takes its first instant"],
)
def test_a_dateless_file_takes_the_time_its_name_spells(
    tmp_path: Path, stem: str, instant: str
) -> None:
    written = build_raster(times=0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / f"{stem}.tif"))

    assert restored.time.values == np.datetime64(instant)


def test_the_datetime_tag_outranks_the_name(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)

    restored = gdal.open_raster(
        geotiff.write_cog(written, tmp_path / "20190507_red.tif")
    )

    assert restored.time.values == instant


def test_several_instants_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one GeoTIFF holds one instant"):
        geotiff.write_cog(build_raster(times=2), tmp_path / "scene.tif")


def test_one_instant_writes_however_the_axis_spells_it(tmp_path: Path) -> None:
    along_an_axis = geotiff.write_cog(build_raster(times=1), tmp_path / "axis.tif")
    as_a_scalar = geotiff.write_cog(
        build_raster(times=1).isel(time=0), tmp_path / "scalar.tif"
    )

    assert gdal.open_raster(along_an_axis).time == gdal.open_raster(as_a_scalar).time


def test_sub_second_time_survives_in_geosave_metadata(tmp_path: Path) -> None:
    instant = np.datetime64("2018-12-26T10:34:39.024000001")
    written = build_raster(times=0).assign_coords(time=instant)

    destination = geotiff.write_cog(written, tmp_path / "20190507_red.tif")
    restored = gdal.open_raster(destination)

    with rasterio.open(destination) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2018:12:26 10:34:39"
        assert src.tags()["GEOSAVE_DATETIME"] == "2018-12-26T10:34:39.024000001"
    assert restored.time.values == instant


def test_the_standard_datetime_tag_remains_the_fallback(tmp_path: Path) -> None:
    path = tmp_path / "20190507_red.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)
        destination.update_tags(TIFFTAG_DATETIME="2024:03:05 06:07:08")

    restored = gdal.open_raster(path)

    assert restored.time.values == np.datetime64("2024-03-05T06:07:08")


def test_the_time_coordinate_overrides_a_carried_datetime_tag(tmp_path: Path) -> None:
    written = (
        build_raster(times=0)
        .assign_coords(time=np.datetime64("2024-03-05T06:07:08"))
        .assign_attrs(TIFFTAG_DATETIME="1999:01:01 00:00:00")
    )

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    with rasterio.open(path) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"


def test_carried_tags_survive_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0)
    written.attrs["TIFFTAG_ARTIST"] = "geosave"

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.attrs["TIFFTAG_ARTIST"] == "geosave"


def test_gtiff_takes_its_own_creation_options(tmp_path: Path) -> None:
    written = build_raster(times=0)

    destination = geotiff.write_gtiff(
        written, tmp_path / "scene.tif", tiled=True, blockxsize=128, blockysize=128
    )

    with rasterio.open(destination) as opened:
        assert opened.block_shapes[0] == (128, 128)


def test_a_chunked_cube_writes_window_by_window(tmp_path: Path) -> None:
    written = build_raster(times=0).chunk({"y": 2, "x": 2})

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    # Streaming is the point; the pixels have to survive it unchanged.
    assert list(restored.data_vars) == list(written.data_vars)
    for name, variable in written.data_vars.items():
        assert np.array_equal(restored[name].values, variable.values)


def test_destination_must_name_a_tiff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        geotiff.write_cog(build_raster(times=0), tmp_path / "scene.nc")


def test_overwrite_guards_an_existing_file(tmp_path: Path) -> None:
    written = build_raster(times=0)
    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    with pytest.raises(FileExistsError, match="pass overwrite=True"):
        geotiff.write_cog(written, destination)

    assert geotiff.write_cog(written, destination, overwrite=True) == destination


def test_an_unnamed_band_keeps_rasterios_own_name(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=2, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2, 2), "uint8"))
        destination.set_band_description(1, "Red reflectance")

    restored = gdal.open_raster(plain)

    # A description is a label, never an identity, so neither band is renamed.
    assert list(restored.data_vars) == ["band_1", "band_2"]
    assert restored["band_1"].attrs["long_name"] == "Red reflectance"


def test_two_bands_naming_one_variable_refuse(tmp_path: Path) -> None:
    clashing = tmp_path / "clashing.tif"
    with rasterio.open(
        clashing, "w", driver="GTiff", height=2, width=2, count=2, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2, 2), "uint8"))
        for band in (1, 2):
            destination.update_tags(band, variable_name="red")

    with pytest.raises(ValueError, match="already names"):
        gdal.open_raster(clashing)


@pytest.mark.parametrize("write", [geotiff.write_cog, geotiff.write_gtiff])
def test_colour_interpretation_round_trips(tmp_path: Path, write) -> None:
    written = build_raster(times=0)
    # uint16 with a role no TIFF photometric holds: the case a creation option
    # silently drops, so GDAL has to carry it in its own metadata instead.
    written = _interpreted(written, red="red", nir="nir")

    destination = write(written, tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        assert [band.name for band in src.colorinterp] == ["red", "nir"]
    restored = gdal.open_raster(destination)
    assert _colorinterp(restored.red) == "red"
    assert _colorinterp(restored.nir) == "nir"


def test_a_cube_measuring_nothing_leaves_gdal_to_interpret_its_bands(
    tmp_path: Path,
) -> None:
    destination = geotiff.write_cog(build_raster(times=0), tmp_path / "scene.tif")

    with rasterio.open(destination) as src:
        assert [band.name for band in src.colorinterp] == ["gray", "undefined"]


def test_colour_interpretation_leaves_a_cog_readable(tmp_path: Path) -> None:
    written = _interpreted(build_raster(times=0), red="red")

    destination = geotiff.write_cog(written, tmp_path / "scene.tif").path

    # A COG keeps its header at the front; reopening one to interpret its bands
    # would move the header to the end and cost the layout.
    with open(destination, "rb") as opened:
        header = opened.read(8)
    endian = "<" if header[:2] == b"II" else ">"
    (first_ifd,) = struct.unpack(f"{endian}I", header[4:8])
    assert first_ifd < destination.stat().st_size / 2
    with rasterio.open(destination) as src:
        assert src.block_shapes[0] != (src.height, src.width)


def test_colour_interpretation_lands_on_each_variable(tmp_path: Path) -> None:
    composite = tmp_path / "rgb.tif"
    with rasterio.open(
        composite, "w", driver="GTiff", height=2, width=2, count=3, dtype="uint8"
    ) as destination:
        destination.write(np.ones((3, 2, 2), "uint8"))
        destination.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
        ]

    restored = gdal.open_raster(composite)

    assert [_colorinterp(restored[name]) for name in sorted(restored.data_vars)] == [
        "red",
        "green",
        "blue",
    ]


def test_an_unreferenced_file_reads_unreferenced(tmp_path: Path) -> None:
    plain = tmp_path / "plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8"
    ) as destination:
        destination.write(np.ones((2, 2), "uint8"), 1)

    assert gdal.open_raster(plain).rio.crs is None


def test_bands_declaring_different_absent_pixels_refuse_to_write(
    tmp_path: Path,
) -> None:
    written = build_raster().gs.write_nodata(0, target="red")

    # GDAL writes one fill value for the file, so rioxarray never sees the mix.
    with pytest.raises(ValueError, match="one GeoTIFF holds one fill value"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_bands_sharing_one_fill_value_write(tmp_path: Path) -> None:
    written = build_raster().gs.write_nodata(0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    assert restored.red.attrs["_FillValue"] == 0
    assert restored.nir.attrs["_FillValue"] == 0


def test_a_units_string_reaches_gdals_own_band_unit(tmp_path: Path) -> None:
    raster = build_raster()
    written = raster.assign(
        {name: raster[name].assign_attrs(units="1") for name in ("red", "nir")}
    )

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    # A tag is text a reader must look for; gdalinfo reads the band's unit.
    with rasterio.open(path) as src:
        assert src.units == ("1", "1")


def test_a_legend_colour_map_writes_a_gdal_palette(tmp_path: Path) -> None:
    labelled = build_raster()[["red"]].astype("uint8")
    written = labelled.gs.write_legend(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )

    path = geotiff.write_cog(written, tmp_path / "labels.tif")

    with rasterio.open(path) as src:
        assert src.colormap(1)[1] == (0, 255, 0, 255)
        # GDAL reads a palette band as palette-interpreted, whatever else it held.
        assert src.colorinterp == (ColorInterp.palette,)


def test_writing_refuses_a_band_count_the_header_does_not_describe(
    tmp_path: Path,
) -> None:
    # A COG refuses reopening for update, so this writes onto the plain GeoTIFF.
    path = geotiff.write_gtiff(build_raster(), tmp_path / "scene.tif")

    with rasterio.open(path, "r+") as dst, pytest.raises(ValueError, match="2 bands"):
        geotiff.write_header(dst, {}, [])


def test_a_legend_keeps_its_class_names_through_a_geotiff(tmp_path: Path) -> None:
    labelled = build_raster()[["red"]].astype("uint8")
    written = labelled.gs.write_legend(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "labels.tif"))

    # A tag holds text, so the class map stays behind but its CF flags travel.
    flags = restored.red.gs.legend
    assert flags.flag_values == [0, 1]
    assert flags.classes == {0: "bg", 1: "palm"}


def test_the_writer_never_states_a_native_value_twice(tmp_path: Path) -> None:
    written = build_raster().gs.write_packing(1e-4)

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    # GDAL holds the scale itself; a metadata item repeating it would say it twice.
    with rasterio.open(path) as src:
        assert src.scales == (1e-4, 1e-4)
        assert "scale_factor" not in src.tags(1)


def test_a_gcp_grid_refuses_to_write_rather_than_lose_its_placing(
    tmp_path: Path,
) -> None:
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

    # A GeoTIFF records this as control points, which this writer does not write.
    with pytest.raises(ValueError, match="GCPGeoBox"):
        geotiff.write_gtiff(gdal.open_raster(source), tmp_path / "out.tif")


def test_absence_kept_in_a_mask_band_says_it_is_unread(tmp_path: Path) -> None:
    source = tmp_path / "masked.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(3e5, 5e6, 10, 10),
    ) as dst:
        dst.write(np.arange(16, dtype="uint8").reshape(1, 4, 4))
        valid = np.full((4, 4), 255, "uint8")
        valid[0, :] = 0
        dst.write_mask(valid)

    # The mask is not one of the bands read, so its absence would vanish silently.
    with pytest.warns(UnreadMaskWarning, match="mask band") as caught:
        gdal.open_raster(source)

    assert Path(caught[0].filename) == Path(__file__)


def test_a_files_own_statistics_are_not_carried_into_memory(tmp_path: Path) -> None:
    path = geotiff.write_cog(build_raster(), tmp_path / "scene.tif")
    subprocess.run(["gdalinfo", "-stats", str(path)], check=True, capture_output=True)

    restored = gdal.open_raster(path)

    # GDAL caches them beside the file; they describe the pixels as they were.
    with rasterio.open(path) as src:
        assert "STATISTICS_MINIMUM" in src.tags(1)
    assert restored.red.gs.statistics().loc["red", "minimum"] is not None
    assert not [key for key in restored.red.attrs if key.startswith("STATISTICS_")]


def test_a_cog_uploads_to_a_remote_location(bucket: str) -> None:
    written = build_raster(times=1).isel(time=0)

    saved = geotiff.write_cog(written, f"{bucket}/scene.tif", storage_options={}).path

    assert saved == UPath(f"{bucket}/scene.tif")
    with rasterio.open(saved.path, opener=saved.fs.open) as src:
        assert src.descriptions == ("red", "nir")


def test_a_palette_outranks_the_interpretation_a_band_was_read_with(
    tmp_path: Path,
) -> None:
    # A band read from a plain GeoTIFF says it is gray; a legend added later colours it.
    plain = geotiff.write_cog(
        build_raster()[["red"]].astype("uint8"), tmp_path / "plain.tif"
    )
    labelled = (
        gdal.open_raster(plain)
        .compute()
        .gs.write_legend(
            {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}, target="red"
        )
    )
    assert labelled.red.attrs["colorinterp"] == "gray"

    restored = gdal.open_raster(geotiff.write_cog(labelled, tmp_path / "labels.tif"))

    colours = restored.red.gs.legend.color_map
    assert colours is not None and colours[1] == "#00FF00"


def test_a_palette_read_from_a_cog_writes_to_netcdf(tmp_path: Path) -> None:
    from geosave.geodata.io import netcdf

    label = (
        build_raster()[["red"]]
        .astype("uint8")
        .gs.write_legend(
            {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}, target="red"
        )
    )
    restored = gdal.open_raster(geotiff.write_cog(label, tmp_path / "labels.tif"))

    # NetCDF attrs hold no mappings, so the palette must be read as the Legend spells it.
    again = netcdf.open_raster(netcdf.write(restored, tmp_path / "labels.nc"))

    assert again.red.gs.legend.color_map == {0: "#000000", 1: "#00FF00"}
