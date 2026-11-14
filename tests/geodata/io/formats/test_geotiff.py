from __future__ import annotations

from tests.conftest import write_cogs


from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from upath import UPath

from geosave.geodata import (
    attrs,
    open_raster,
)
from geosave.geodata.io import rasterio as gdal
from geosave.geodata.io.formats import geotiff
from geosave.geodata.warnings import UnreadMaskWarning
from tests.conftest import place
from tests.geodata.conftest import (
    _refuse,
    assign_cells,
    build_raster,
)


from geosave.geodata import io
from geosave.geodata.io.formats import netcdf


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
    with gdal.open_raster(destination) as restored:
        assert restored[first].attrs["long_name"] == written[first].attrs["long_name"]


def test_an_instant_is_stated_by_the_tiff_datetime_tag(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)
    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    restored = gdal.open_raster(path)

    with rasterio.open(path) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2024:03:05 06:07:08"
        # An instant covers no period.
        assert "time_coverage_start" not in src.tags()
    assert restored.time.values == instant
    # The date lives on the time coordinate alone, so dates join without conflict.
    assert "TIFFTAG_DATETIME" not in restored.attrs
    assert "start_datetime" not in restored.coords


def test_a_cog_keeps_its_time_and_the_period_it_covers(tmp_path: Path) -> None:
    passed = np.datetime64("2025-06-01T10:30:00")
    day = (np.datetime64("2025-06-01T00:00:00"), np.datetime64("2025-06-02T00:00:00"))
    written = assign_cells(build_raster(times=0).assign_coords(time=passed), [day])
    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    restored = gdal.open_raster(path)

    with rasterio.open(path) as src:
        assert src.tags()["TIFFTAG_DATETIME"] == "2025:06:01 10:30:00"
        assert src.tags()["time_coverage_start"] == "2025-06-01"
        assert src.tags()["time_coverage_end"] == "2025-06-02"
    assert restored.time.values == passed
    assert restored.start_datetime.values == day[0]
    assert restored.end_datetime.values == day[1]


def test_opening_a_file_reads_no_pixels(tmp_path: Path) -> None:
    path = geotiff.write_cog(build_raster(times=0), tmp_path / "scene_20240305.tif")

    with dask.config.set(scheduler=_refuse):
        restored = gdal.open_raster(path)

    assert all(isinstance(restored[name].data, da.Array) for name in restored.data_vars)


@pytest.mark.parametrize(
    ("stem", "instant"),
    [
        ("20190507_red", "2019-05-07T00:00:00"),
        ("20250601T000000_red", "2025-06-01T00:00:00"),
    ],
    ids=["a day", "an instant"],
)
def test_a_dateless_file_takes_the_time_its_name_spells(
    tmp_path: Path, stem: str, instant: str
) -> None:
    written = build_raster(times=0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / f"{stem}.tif"))

    assert restored.time.values == np.datetime64(instant)


def test_several_instants_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one GeoTIFF holds one instant"):
        geotiff.write_cog(build_raster(times=2), tmp_path / "scene.tif")


def test_a_chunked_cube_writes_window_by_window(tmp_path: Path) -> None:
    written = build_raster(times=0).chunk({"y": 2, "x": 2})

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    # Streaming is the point; the pixels have to survive it unchanged.
    assert list(restored.data_vars) == list(written.data_vars)
    for name, variable in written.data_vars.items():
        assert np.array_equal(restored[name].values, variable.values)


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


def test_external_rgb_reads_and_rewrites_the_display(tmp_path: Path) -> None:
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

    assert restored.gs.rgb == ("band_1", "band_2", "band_3")
    # The file has no grid: supply one explicitly before writing the composite.
    restored = restored.pipe(place, build_raster().gs.geobox)
    path = geotiff.write_cog(restored, tmp_path / "rewritten.tif")
    with rasterio.open(path) as saved:
        assert saved.colorinterp == (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
        )
    with open_raster(path) as reopened:
        assert reopened.gs.rgb == restored.gs.rgb
        for name in restored.data_vars:
            np.testing.assert_array_equal(reopened[name], restored[name])


def test_bands_declaring_different_absent_pixels_refuse_to_write(
    tmp_path: Path,
) -> None:
    written = build_raster().gs.write_nodata(0, target=["red"])

    # GDAL writes one fill value for the file, so rioxarray never sees the mix.
    with pytest.raises(ValueError, match="one GeoTIFF holds one fill value"):
        geotiff.write_cog(written, tmp_path / "scene.tif")


def test_a_legend_colour_map_writes_a_gdal_palette(tmp_path: Path) -> None:
    labelled = build_raster()[["red"]].astype("uint8")
    written = labelled.gs.write_flags(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )

    path = geotiff.write_cog(written, tmp_path / "labels.tif")

    with rasterio.open(path) as src:
        assert src.colormap(1)[1] == (0, 255, 0, 255)
        # GDAL reads a palette band as palette-interpreted, whatever else it held.
        assert src.colorinterp == (ColorInterp.palette,)


def test_the_writer_never_states_a_native_value_twice(tmp_path: Path) -> None:
    written = build_raster().gs.write_packing(attrs.CFPacking(scale_factor=1e-4))

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    # GDAL holds the scale itself; a metadata item repeating it would say it twice.
    with rasterio.open(path) as src:
        assert src.scales == (1e-4, 1e-4)
        assert "scale_factor" not in src.tags(1)


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


def test_a_cog_uploads_to_a_remote_location(bucket: str) -> None:
    written = build_raster(times=1).isel(time=0)

    saved = geotiff.write_cog(written, f"{bucket}/scene.tif", storage_options={})

    assert saved == UPath(f"{bucket}/scene.tif")
    with rasterio.open(saved.path, opener=saved.fs.open) as src:
        assert src.descriptions == ("red", "nir")


def test_a_palette_read_from_a_cog_writes_to_netcdf(tmp_path: Path) -> None:

    label = (
        build_raster()[["red"]]
        .astype("uint8")
        .gs.write_flags(
            {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}, target=["red"]
        )
    )
    restored = gdal.open_raster(geotiff.write_cog(label, tmp_path / "labels.tif"))

    # NetCDF attrs hold no list of strings, so the hints must be read as CFFlags spells them.
    again = netcdf.open_netcdf(netcdf.write_netcdf(restored, tmp_path / "labels.nc"))

    assert again.red.gs.flags.color_hints == ["000000", "00FF00"]


@pytest.mark.integration
def test_cogs_round_trip_lazily(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)

    paths = write_cogs(
        source, prefix, format_paths="forest_{time:%Y%m%d}.tif", storage_options=options
    )

    assert all(str(path).startswith(prefix) for path in paths)
    for index, path in enumerate(paths):
        with dask.config.set(scheduler=_refuse):
            restored = open_raster(path, chunks={})
        np.testing.assert_array_equal(
            restored.red.values, source.red.isel(time=index).values
        )
        restored.close()


def test_each_raster_is_written_to_its_destination(tmp_path):
    files = {
        UPath(tmp_path / "red.tif"): build_raster()[["red"]],
        UPath(tmp_path / "nir.tif"): build_raster()[["nir"]],
    }

    written = geotiff.write_files(files)

    assert [path.name for path in written] == ["red.tif", "nir.tif"]
    assert all(path.is_file() for path in written)


def test_a_deferred_batch_waits_for_compute(tmp_path):
    files = {UPath(tmp_path / "red.tif"): build_raster()[["red"]].chunk()}

    pending = geotiff.write_files(files, compute=False)

    assert not list(tmp_path.iterdir())
    assert [path.name for path in pending.compute()] == ["red.tif"]


def test_a_later_invalid_raster_prevents_every_write(tmp_path):
    compute = True
    files = {
        UPath(tmp_path / "a.tif"): build_raster(times=1),
        UPath(tmp_path / "b.tif"): build_raster(times=2),
    }

    with pytest.raises(ValueError, match="one instant"):
        geotiff.write_files(files, compute=compute)

    assert list(tmp_path.iterdir()) == []


def test_an_existing_later_destination_prevents_earlier_writes(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"keep me")
    files = {
        UPath(tmp_path / "a.tif"): build_raster(),
        UPath(tmp_path / "b.tif"): build_raster(),
    }

    with pytest.raises(FileExistsError):
        geotiff.write_files(files)

    assert not (tmp_path / "a.tif").exists()
    assert (tmp_path / "b.tif").read_bytes() == b"keep me"


def test_nodata_only_decoding_restores_integer_geotiff_storage(tmp_path):
    raw = build_raster().gs.write_nodata(65535)
    raw.red.data = np.array([[0, 1000], [2000, 65535]], "uint16")
    decoded = raw.gs.mask_and_scale()
    path = tmp_path / "nodata.tif"

    geotiff.write_cog(decoded, path)
    with io.open_raster(path) as stored:
        assert stored.red.dtype == raw.red.dtype
        assert stored.red.gs.nodata == 65535
        np.testing.assert_array_equal(stored.red, raw.red)


@pytest.mark.parametrize("state", ["raw", "decoded", "derived"])
def test_geotiff_encodes_each_band_once(tmp_path, state):
    import rasterio
    from dask.callbacks import Callback

    raw = build_raster(times=0).gs.write_nodata(0)
    raw.red.attrs.update(scale_factor=0.01, add_offset=0.5)
    raw.nir.attrs.update(scale_factor=0.02, add_offset=1.0)
    expected = xr.decode_cf(raw, decode_coords=False)
    source = raw
    if state == "decoded":
        source = expected
    elif state == "derived":
        source = expected * 2
        expected = source.copy(deep=True)
        for name in source.data_vars:
            source[name].encoding = {
                "dtype": "uint16",
                "_FillValue": 0,
                "scale_factor": 0.01,
                "add_offset": 0.0,
            }
    source = source.chunk({"x": 1, "y": 1})
    before = source.copy(deep=True)
    destination = tmp_path / "bands.tif"

    with Callback(pretask=lambda *_: pytest.fail("planning read pixels")):
        pending = geotiff.write_cog(source, destination, compute=False)
    assert not destination.exists()
    pending.compute()

    with io.open_raster(destination, mask_and_scale=True) as reopened:
        assert reopened.gs.geobox == source.gs.geobox
        for name in source.data_vars:
            np.testing.assert_allclose(reopened[name], expected[name], equal_nan=True)
    with rasterio.open(destination) as stored:
        assert stored.dtypes == ("uint16", "uint16")
        assert stored.nodata == 0
        assert stored.scales == ((0.01, 0.01) if state == "derived" else (0.01, 0.02))
        assert stored.offsets == ((0.0, 0.0) if state == "derived" else (0.5, 1.0))
    xr.testing.assert_identical(source, before)
    for name in source.data_vars:
        assert source[name].encoding == before[name].encoding


@pytest.mark.parametrize("tiff_tags", [False, True])
def test_acdd_attrs_round_trip_with_or_without_tiff_tags(tmp_path, tiff_tags) -> None:
    import re

    from geosave.geodata import open_raster

    source = build_raster().assign_attrs(title="Forest", license="CC-BY-4.0")
    path = geotiff.write_cog(source, tmp_path / "scene.tif", tiff_tags=tiff_tags)

    with rasterio.open(path) as src:
        tags = src.tags()
    xml = re.search(rb"<GDALMetadata>.*?</GDALMetadata>", path.read_bytes(), re.S)
    in_gdal_xml = xml is not None and b"Forest" in xml.group(0)
    assert in_gdal_xml is not tiff_tags
    assert ("TIFFTAG_DOCUMENTNAME" in tags) is tiff_tags
    assert ("TIFFTAG_SOFTWARE" in tags) is tiff_tags
    restored = open_raster(path)
    assert restored.attrs["title"] == "Forest"
    assert restored.attrs["license"] == "CC-BY-4.0"


@pytest.mark.parametrize("writer", [geotiff.write_cog, geotiff.write_geotiff])
def test_single_write_rechecks_destination_when_computed(writer, tmp_path):
    path = tmp_path / "later.tif"
    pending = writer(build_raster(times=0), path, compute=False)
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        pending.compute(scheduler="synchronous")
    assert path.read_bytes() == b"keep me"


def test_deferred_batch_rechecks_all_destinations_before_writing(tmp_path):
    first, second = UPath(tmp_path / "a.tif"), UPath(tmp_path / "b.tif")
    pending = geotiff.write_files(
        {first: build_raster(times=0), second: build_raster(times=0)}, compute=False
    )
    second.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        pending.compute(scheduler="synchronous")
    assert not first.exists()
    assert second.read_bytes() == b"keep me"


@pytest.mark.parametrize("writer", [geotiff.write_cog, geotiff.write_geotiff])
def test_nested_creation_options_reach_gdal(writer, tmp_path):
    path = writer(
        build_raster(times=0),
        tmp_path / "image.tif",
        creation_options={"COMPRESS": "LZW"},
    )
    with rasterio.open(path) as stored:
        assert stored.compression.name == "lzw"


@pytest.mark.parametrize(
    "extra",
    [
        {"COMPRESS": "LZW"},
        {"DRIVER": "PNG"},
        {"DTYPE": "uint8"},
        {"LEVEL": 1, "level": 2},
    ],
)
def test_nested_options_cannot_override_direct_or_managed_options(extra, tmp_path):
    path = tmp_path / "image.tif"
    with pytest.raises(TypeError, match="creation_options"):
        geotiff.write_cog(
            build_raster(times=0), path, compress="DEFLATE", creation_options=extra
        )
    assert not path.exists()


@pytest.mark.parametrize("compute", [True, False])
def test_invalid_later_time_cells_prevent_all_batch_writes(tmp_path, compute):
    first = build_raster(times=0)
    broken = first.assign_coords(
        time=np.datetime64("2025-01-01"), start_datetime=np.datetime64("2025-01-01")
    )
    with pytest.raises(KeyError, match="end_datetime"):
        geotiff.write_files(
            {UPath(tmp_path / "a.tif"): first, UPath(tmp_path / "b.tif"): broken},
            compute=compute,
        )
    assert not list(tmp_path.iterdir())


def test_cog_destination_can_be_named_source(tmp_path):
    path = geotiff.write_cog(build_raster(times=0), tmp_path / "source.tif")
    with rasterio.open(path) as stored:
        assert stored.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
