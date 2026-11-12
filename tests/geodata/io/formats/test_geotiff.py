from __future__ import annotations

from tests.conftest import write_cogs

from tests.conftest import combine_item

import struct
import subprocess
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from upath import UPath

import geosave as gs
from geosave.geodata import (
    attrs,
    open_raster,
)
from geosave.geodata.io import rasterio as gdal
from geosave.geodata.io.formats import geotiff
from geosave.geodata.warnings import UnreadMaskWarning
from tests.conftest import build_dataset, place
from tests.geodata.conftest import (
    _refuse,
    assign_cells,
    build_raster,
    create_file_items,
    create_test_collection,
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


def test_a_band_described_by_its_name_gains_no_long_name(tmp_path: Path) -> None:
    # rioxarray describes an unlabelled band by its variable name.
    written = build_raster(times=0)

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    for name in written.data_vars:
        assert "long_name" not in restored[name].attrs


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


def test_grouped_time_bounds_survive_cog_persistence(tmp_path: Path) -> None:
    dataset = build_raster(times=2)
    bounds = np.array(
        [["2024-01-01", "2024-01-03"], ["2024-02-01", "2024-02-03"]],
        dtype="datetime64[ns]",
    )
    dataset = dataset.assign_coords(time_bounds=(("time", "edge"), bounds))
    dataset = assign_cells(dataset, list(map(tuple, bounds)))

    for index, (_, observation) in enumerate(dataset.groupby("time")):
        observation = observation.squeeze("time", drop=False)
        path = geotiff.write_cog(observation, tmp_path / f"{index}.tif")

        with rasterio.open(path) as raster:
            assert raster.tags()["time_coverage_start"] == np.datetime_as_string(
                bounds[index, 0].astype("datetime64[us]"), unit="auto"
            )
            assert raster.tags()["time_coverage_end"] == np.datetime_as_string(
                bounds[index, 1].astype("datetime64[us]"), unit="auto"
            )
            np.testing.assert_array_equal(raster.read(1), observation.red.values)
            assert raster.transform == observation.gs.geobox.transform


@pytest.mark.parametrize("dated", [False, True])
def test_opening_a_file_reads_no_pixels(tmp_path: Path, dated: bool) -> None:
    name = "scene_20240305.tif" if dated else "scene.tif"
    path = geotiff.write_cog(build_raster(times=0), tmp_path / name)

    with dask.config.set(scheduler=_refuse):
        restored = gdal.open_raster(path)

    assert all(isinstance(restored[name].data, da.Array) for name in restored.data_vars)


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
    instant = np.datetime64("2024-03-05T06:07:08")
    written = (
        build_raster(times=0)
        .assign_coords(time=instant)
        .assign_attrs(TIFFTAG_DATETIME="1999:01:01 00:00:00")
    )

    path = geotiff.write_cog(written, tmp_path / "scene.tif")

    assert gdal.open_raster(path).time.values == instant


def test_carried_tags_survive_the_round_trip(tmp_path: Path) -> None:
    written = build_raster(times=0)
    written.attrs["TIFFTAG_ARTIST"] = "geosave"

    restored = gdal.open_raster(geotiff.write_cog(written, tmp_path / "scene.tif"))

    # A baseline TIFF tag reads back as the ACDD attr it carries.
    assert restored.attrs["institution"] == "geosave"


def test_gtiff_takes_its_own_creation_options(tmp_path: Path) -> None:
    written = build_raster(times=0)

    destination = geotiff.write_geotiff(
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


def test_colour_interpretation_leaves_a_cog_readable(tmp_path: Path) -> None:
    written = build_raster(times=0)
    written["green"] = written.red.copy()
    written = written.gs.write_rgb("red", "green", "nir")

    destination = geotiff.write_cog(written, tmp_path / "scene.tif")

    # A COG keeps its header at the front; reopening one to interpret its bands
    # would move the header to the end and cost the layout.
    with open(destination, "rb") as opened:
        header = opened.read(8)
    endian = "<" if header[:2] == b"II" else ">"
    (first_ifd,) = struct.unpack(f"{endian}I", header[4:8])
    assert first_ifd < destination.stat().st_size / 2
    with rasterio.open(destination) as src:
        assert src.block_shapes[0] != (src.height, src.width)


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
    written = build_raster().gs.write_nodata(0, target=["red"])

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

    # Rasters placed by control points are not supported yet.
    with pytest.raises(NotImplementedError, match="ground control points"):
        geotiff.write_geotiff(gdal.open_raster(source), tmp_path / "out.tif")


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

    saved = geotiff.write_cog(written, f"{bucket}/scene.tif", storage_options={})

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
        .gs.write_flags(
            {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}, target=["red"]
        )
    )

    restored = gdal.open_raster(geotiff.write_cog(labelled, tmp_path / "labels.tif"))

    colours = restored.red.gs.display.color_map
    assert colours is not None and colours[1] == "#00FF00"


def test_a_palette_read_from_a_cog_writes_to_netcdf(tmp_path: Path) -> None:

    label = (
        build_raster()[["red"]]
        .astype("uint8")
        .gs.write_flags(
            {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}, target=["red"]
        )
    )
    restored = gdal.open_raster(geotiff.write_cog(label, tmp_path / "labels.tif"))

    # NetCDF attrs hold no mappings, so the palette must be read as Display spells it.
    again = netcdf.open_netcdf(netcdf.write_netcdf(restored, tmp_path / "labels.nc"))

    assert again.red.gs.display.color_map == {0: "#000000", 1: "#00FF00"}


@pytest.mark.parametrize(
    ("explicit", "period"),
    [(False, ("2024-02-01", "2024-03-01")), (True, ("2024-02-01", "2024-03-01"))],
    ids=["bin", "explicit"],
)
def test_saved_paths_preserve_composite_coverage(
    tmp_path: Path, explicit: bool, period: tuple[str, str]
):
    grid = GeoBox.from_bbox((0, 0, 20, 20), "EPSG:3857", resolution=10)
    source = build_dataset(
        {"red": (("time", "y", "x"), da.full((2, 2, 2), 5.0, chunks=(1, 2, 2)))},
        grid,
        time=pd.to_datetime(["2024-02-05", "2024-02-20"]),
        nodata=np.nan,
    )
    monthly = source.gs.mask_and_scale().gs.composite("MS", "median")
    if explicit:
        monthly = assign_cells(
            monthly, [(np.datetime64("2024-02-01"), np.datetime64("2024-03-01"))]
        )
    write_cogs(monthly, tmp_path, format_paths="{time:%Y%m}.tif")
    collection = create_test_collection("image", description="Monthly imagery")
    items = list(
        create_file_items({"image": tmp_path / "202402.tif"}, collection=collection)
    )
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    assert pd.Timestamp(items[0].properties["start_datetime"]) == start.tz_localize(
        "UTC"
    )
    assert pd.Timestamp(items[0].properties["end_datetime"]) == end.tz_localize("UTC")
    from geosave.geodata.stac.catalog import create_catalog

    table = create_catalog(items, collections=[collection])
    restored = gs.open_vector(table.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    raster = combine_item(restored.iloc[0].gs.to_datatree())
    assert raster.to_array().chunks is not None
    assert raster.gs.timespan == (start, end)
    np.testing.assert_allclose(raster.to_array().values, 5)


@pytest.mark.integration
def test_cogs_round_trip_lazily(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)

    paths = write_cogs(
        source, prefix, format_paths="forest_{time:%Y%m%d}.tif", storage_options=options
    )

    assert all(str(path).startswith(prefix) for path in paths)
    with dask.config.set(scheduler=_refuse):
        restored = open_raster(list(paths), chunks={})
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


@pytest.mark.parametrize("writer", [geotiff.write_cog, geotiff.write_geotiff])
def test_direct_writer_preserves_explicit_filesystem_options(writer, bucket):
    from tests.geodata.conftest import build_raster

    source = build_raster(packed=True)
    saved = writer(
        source,
        f"{bucket}/explicit-options.tif",
        storage_options={"skip_instance_cache": True},
    )
    assert saved.storage_options.get("skip_instance_cache") is True
    with open_raster(saved) as restored:
        assert restored.gs.geobox == source.gs.geobox
        np.testing.assert_array_equal(restored.red, source.red)


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


@pytest.mark.parametrize("compute", [True, False])
def test_a_later_invalid_raster_prevents_every_write(tmp_path, compute):
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


def test_a_deferred_batch_rechecks_its_destinations_when_computed(tmp_path):
    files = {
        UPath(tmp_path / "a.tif"): build_raster(),
        UPath(tmp_path / "b.tif"): build_raster(),
    }
    pending = geotiff.write_files(files, compute=False)
    (tmp_path / "b.tif").write_bytes(b"created after planning")

    with pytest.raises(FileExistsError):
        pending.compute()

    assert not (tmp_path / "a.tif").exists()


def test_two_destinations_resolving_to_one_file_collide(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = {
        UPath("same.tif"): build_raster(),
        UPath(tmp_path / "same.tif"): build_raster(),
    }

    with pytest.raises(ValueError, match="collide"):
        geotiff.write_files(files)
    assert not list(tmp_path.iterdir())


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


@pytest.mark.parametrize("writer", [geotiff.write_cog, geotiff.write_geotiff])
@pytest.mark.parametrize("state", ["raw", "decoded", "mixed", "derived"])
@pytest.mark.parametrize("lazy", [False, True])
def test_geotiff_encodes_each_band_once(tmp_path, writer, state, lazy):
    import rasterio
    from dask.callbacks import Callback

    raw = build_raster(times=0).gs.write_nodata(0)
    raw.red.attrs.update(scale_factor=0.01, add_offset=0.5)
    raw.nir.attrs.update(scale_factor=0.02, add_offset=1.0)
    expected = xr.decode_cf(raw, decode_coords=False)
    source = raw
    if state == "decoded":
        source = expected
    elif state == "mixed":
        source = raw.assign(red=expected.red)
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
    if lazy:
        source = source.chunk({"x": 1, "y": 1})
    before = source.copy(deep=True)
    destination = tmp_path / "bands.tif"

    with Callback(pretask=lambda *_: pytest.fail("planning read pixels")):
        pending = writer(source, destination, compute=False)
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


def test_an_extra_nonspatial_axis_cannot_be_stored_as_one_cog(tmp_path):
    source = build_raster().expand_dims(model=["a", "b"])
    with pytest.raises(ValueError, match="one instant of one grid"):
        geotiff.write_cog(source, tmp_path / "scene.tif")
    assert not (tmp_path / "scene.tif").exists()
