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
from geosave.geodata.io.raster import geotiff as gdal
from geosave.geodata.io.raster.geotiff import hf_to_s3
from geosave.geodata.io.raster import geotiff
from geosave.geodata.warnings import UnreadMaskWarning
from tests.conftest import place
from tests.geodata.conftest import (
    _refuse,
    assign_cells,
    build_raster,
)


from geosave.geodata import io
from geosave.geodata.io.raster import netcdf


def test_round_trip_keeps_the_variable_names(tmp_path: Path) -> None:
    written = build_raster(times=0)

    restored = gdal.open_geotiff(geotiff.write_cog(written, tmp_path / "scene.tif"))

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
    with gdal.open_geotiff(destination) as restored:
        assert restored[first].attrs["long_name"] == written[first].attrs["long_name"]


def test_an_instant_is_stated_by_the_tiff_datetime_tag(tmp_path: Path) -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    written = build_raster(times=0).assign_coords(time=instant)
    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    restored = gdal.open_geotiff(path)

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

    restored = gdal.open_geotiff(path)

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
        restored = gdal.open_geotiff(path)

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

    restored = gdal.open_geotiff(geotiff.write_cog(written, tmp_path / f"{stem}.tif"))

    assert restored.time.values == np.datetime64(instant)


def test_several_instants_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one GeoTIFF holds one instant"):
        geotiff.write_cog(build_raster(times=2), tmp_path / "scene.tif")


def test_a_chunked_cube_writes_window_by_window(tmp_path: Path) -> None:
    written = build_raster(times=0).chunk({"y": 2, "x": 2})

    restored = gdal.open_geotiff(geotiff.write_cog(written, tmp_path / "scene.tif"))

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

    restored = gdal.open_geotiff(plain)

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

    restored = gdal.open_geotiff(composite)

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

    # A GeoTIFF holds one band stack, and stacked pixels share one meaning.
    with pytest.raises(ValueError, match="different CF packing"):
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
        gdal.open_geotiff(source)

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
    restored = gdal.open_geotiff(geotiff.write_cog(label, tmp_path / "labels.tif"))

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
    raw.nir.attrs.update(scale_factor=0.01, add_offset=0.5)
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
        assert stored.scales == (0.01, 0.01)
        assert stored.offsets == ((0.0, 0.0) if state == "derived" else (0.5, 0.5))
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
def test_creation_options_reach_gdal(writer, tmp_path):
    path = writer(
        build_raster(times=0),
        tmp_path / "image.tif",
        compress="LZW",
    )
    with rasterio.open(path) as stored:
        assert stored.compression.name == "lzw"


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


@pytest.mark.parametrize("writer", [geotiff.write_geotiff, geotiff.write_cog])
def test_writing_computes_each_source_chunk_once(writer, tmp_path):
    from dask import delayed

    reads = []

    @delayed
    def block(value):
        reads.append(value)
        return np.full((32, 32), value, dtype="uint16")

    pixels = da.block(
        [
            [
                da.from_delayed(block(value), shape=(32, 32), dtype="uint16")
                for value in row
            ]
            for row in ((1, 2), (3, 4))
        ]
    )
    source = (
        xr.DataArray(
            pixels,
            dims=("y", "x"),
            coords={"y": 1000 - np.arange(64), "x": np.arange(64)},
            name="red",
        )
        .rio.write_crs("EPSG:32649")
        .to_dataset()
    )
    with dask.config.set(scheduler="synchronous"):
        path = writer(source, tmp_path / "chunks.tif")
    assert sorted(reads) == [1, 2, 3, 4]
    with rasterio.open(path) as stored:
        np.testing.assert_array_equal(stored.read(1)[::32, ::32], [[1, 2], [3, 4]])


@pytest.mark.parametrize("writer", [geotiff.write_geotiff, geotiff.write_cog])
def test_filesystem_opener_round_trip_without_download(writer, tmp_path, monkeypatch):
    path = UPath(f"memory://{tmp_path.name}/scene.tif")
    source = build_raster(times=0).gs.write_nodata(0)
    source.red.attrs.update(scale_factor=0.01, add_offset=0.5)
    source.nir.attrs.update(scale_factor=0.01, add_offset=0.5)
    writer(source, path)
    monkeypatch.setattr(
        path.fs, "get_file", lambda *a, **kw: pytest.fail("downloaded file")
    )
    try:
        with io.open_raster(path, mask_and_scale=True) as opened:
            expected = xr.decode_cf(source, decode_coords=False)
            for name in source.data_vars:
                np.testing.assert_allclose(opened[name], expected[name], equal_nan=True)
            rewritten = tmp_path / "rewritten.tif"
            geotiff.write_geotiff(opened, rewritten)
        with io.open_raster(rewritten) as raw:
            for name in source.data_vars:
                np.testing.assert_array_equal(raw[name], source[name])
                assert raw[name].dtype == source[name].dtype
    finally:
        path.unlink()


@pytest.mark.parametrize("writer", [geotiff.write_geotiff, geotiff.write_cog])
def test_a_band_stack_reads_back_as_written(writer, tmp_path):
    written = build_raster(times=1).gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4, fill_value=0)
    )
    written = written.assign(blue=written.red)
    written = attrs.stamp_attrs(written, attrs.Display(rgb=["red", "nir", "blue"]))
    written = written.assign_attrs(title="Forest")
    written = written.assign_coords(
        start_datetime=("time", written.time.values),
        end_datetime=("time", written.time.values + np.timedelta64(1, "D")),
    )
    for name in written.data_vars:
        written[name].attrs["units"] = "1"
    written.red.attrs.update(long_name="Red reflectance", wavelength="0.665")

    restored = io.open_raster(writer(written, tmp_path / "scene.tif"))

    # A GeoTIFF holds one instant, so its time dim comes back as scalar coords.
    expected = written.squeeze("time")
    assert list(restored.data_vars) == list(expected.data_vars)
    assert restored.gs.geobox == expected.gs.geobox
    assert restored.time.values == expected.time.values
    assert restored.start_datetime.values == expected.start_datetime.values
    assert restored.end_datetime.values == expected.end_datetime.values
    # GDAL adds the pixel-area convention to the file's own tags.
    assert restored.attrs.pop("AREA_OR_POINT") == "Area"
    assert restored.attrs == expected.attrs
    for name in expected.data_vars:
        assert restored[name].attrs == expected[name].attrs
        assert restored[name].dtype == expected[name].dtype
        np.testing.assert_array_equal(restored[name], expected[name])


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("hf://buckets/me/samples/forest/a.tif", "s3://me/samples/forest/a.tif"),
        ("/data/a.tif", "/data/a.tif"),
    ],
)
def test_gdal_reads_an_hf_bucket_through_its_s3_gateway(location, expected) -> None:
    assert hf_to_s3(location) == expected


def _tiff(path, *, tiled: bool) -> str:
    import rasterio
    from rasterio.transform import from_origin

    options = {"tiled": True, "blockxsize": 256, "blockysize": 256} if tiled else {}
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=1,
        dtype="uint16",
        crs="EPSG:32749",
        transform=from_origin(500_000.0, 9_010_240.0, 10, 10),
        **options,
    ) as tiff:
        tiff.write(np.zeros((1, 1024, 1024), "uint16"))
    return str(path)


def test_lazy_chunks_follow_the_files_own_tiles(tmp_path) -> None:
    # A window then reads only the tiles it overlaps, not the whole band.
    from geosave.geodata.io.raster.geotiff import open_geotiff as open_raster

    band = open_raster(_tiff(tmp_path / "tiled.tif", tiled=True), chunks={}).band_1

    assert band.data.chunksize == (256, 256)


def _refuse_to_open(monkeypatch) -> list:
    """Record what rasterio is asked to open, opening nothing."""
    import rasterio

    opened = []

    def spy(fp, *args, **kwargs):
        opened.append(str(fp))
        raise rasterio.errors.RasterioIOError("no network in tests")

    monkeypatch.setattr(rasterio, "open", spy)
    return opened


def test_an_hf_bucket_without_options_keeps_its_s3_gateway(monkeypatch) -> None:
    import rasterio

    from geosave.geodata.io.raster.geotiff import open_geotiff as open_raster

    opened = _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("hf://buckets/me/samples/a.tif")

    assert opened == ["s3://me/samples/a.tif"]


def test_each_band_reads_its_own_pixels(tmp_path: Path) -> None:
    import xarray as xr

    from geosave.geodata.io.raster.geotiff import open_geotiff as open_raster

    values = np.stack([np.full((4, 4), k, "uint16") for k in (10, 20, 30)])
    cube = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={
            "band": [1, 2, 3],
            "y": 9_000_000 - 10 * np.arange(4),
            "x": 500_000 + 10 * np.arange(4),
        },
    ).rio.write_crs("EPSG:32749")
    cube.rio.to_raster(tmp_path / "three.tif")

    opened = open_raster(tmp_path / "three.tif")

    for index, variable in enumerate(opened.data_vars.values()):
        np.testing.assert_array_equal(variable.values, values[index])


@pytest.mark.parametrize("duplicate_names", [False, True])
def test_reader_closes_native_backend_on_close_or_mapping_error(
    tmp_path, monkeypatch, duplicate_names
):
    import rasterio
    import xarray as xr
    from geosave.geodata.io.raster import geotiff as reader

    path = _tiff(tmp_path / "reader.tif", tiled=False)
    # One native backend handle for this test; metadata still comes from a real TIFF.
    closed = []
    opened = xr.DataArray(np.zeros((1, 1, 1)), dims=("band", "y", "x"))
    if duplicate_names:
        # Two bands declaring the same variable must close the backend on failure.
        with rasterio.open(path) as src:
            profile = src.profile
        profile["count"] = 2
        with rasterio.open(path, "w", **profile) as dst:
            dst.update_tags(1, variable_name="same")
            dst.update_tags(2, variable_name="same")
        opened = xr.concat([opened, opened], dim="band")
    opened.set_close(lambda: closed.append(True))
    monkeypatch.setattr(reader.rioxarray, "open_rasterio", lambda *a, **kw: opened)
    if duplicate_names:
        with pytest.raises(ValueError, match="duplicate"):
            reader.open_geotiff(path)
    else:
        result = reader.open_geotiff(path)
        assert closed == []
        result.close()
        result.close()
    assert closed == [True]
