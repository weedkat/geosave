"""A raster arranged as files by a path template, and written through a writer."""

from __future__ import annotations

from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import rasterio

from geosave.geodata import open_raster
from geosave.geodata.io.raster import layout
from geosave.geodata.io.raster.file import RasterFile
from upath import UPath
from tests.geodata.conftest import build_raster, flat

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

    files = flat(written.gs.to_cog(f"{tmp_path}/{template}"))

    assert [Path(f) for f in files] == [tmp_path / name for name in expected]
    restored = open_raster(files)
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        np.testing.assert_array_equal(restored[name].squeeze(), variable.squeeze())


def test_place_is_a_callers_field_not_a_built_in(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="lat"):
        flat(build_raster().gs.to_cog(f"{tmp_path}/{{lat:.3f}}.tif"))


def test_several_dates_without_time_refuse(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match=r"2 dates would share .*scene\.tif; add \{time\}"
    ):
        flat(build_raster(times=2).gs.to_cog(f"{tmp_path}/scene.tif"))
    assert not list(tmp_path.iterdir())


def test_a_partial_date_that_collides_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"make \{time:%Y\} more precise"):
        flat(build_raster(times=2).gs.to_cog(f"{tmp_path}/{{time:%Y}}.tif"))


def test_time_on_an_undated_raster_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no dates, so \{time\} cannot be filled"):
        flat(build_raster().gs.to_cog(f"{tmp_path}/{{time:%Y}}.tif"))


def test_a_path_that_is_not_a_tiff_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names no raster format"):
        flat(build_raster().gs.to_cog(f"{tmp_path}/scene"))


def test_sub_second_instants_render_through_strftime(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])

    (file,) = flat(written.gs.to_cog(f"{tmp_path}/{{time:%Y%m%dT%H%M%S_%f}}.tif"))

    assert file.name == "20250601T103031_123456.tif"
    assert open_raster(file).time.values == instant


def test_joined_bands_keep_the_variable_order(tmp_path: Path) -> None:
    written = build_raster(times=2)

    paths = flat(written.gs.to_cog(f"{tmp_path}/{{time:%Y%m%d}}.tif"))

    assert list(open_raster(paths).data_vars) == list(written.data_vars)


def test_map_scale_reaches_every_file(tmp_path: Path) -> None:
    source = build_raster(times=2)
    written = source.assign(
        {
            name: source[name].assign_attrs(units="reflectance")
            for name in ("red", "nir")
        }
    )

    files = flat(written.gs.to_cog(f"{tmp_path}/{{time:%Y%m%d}}.tif", map_scale=10_000))

    for file in files:
        with rasterio.open(file) as src:
            assert float(src.tags()["TIFFTAG_XRESOLUTION"]) == pytest.approx(10.0)
    assert open_raster(files).red.attrs["units"] == "reflectance"


def test_an_existing_file_refuses_without_overwrite(tmp_path: Path) -> None:
    written = build_raster(times=1)
    flat(written.gs.to_cog(f"{tmp_path}/scene.tif"))

    with pytest.raises(FileExistsError):
        flat(written.gs.to_cog(f"{tmp_path}/scene.tif"))
    assert flat(written.gs.to_cog(f"{tmp_path}/scene.tif", overwrite=True))


def test_a_raster_spanning_more_than_time_refuses(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a COG holds one instant of one grid"):
        flat(spanning.gs.to_cog(f"{tmp_path}/{{time:%Y%m%d}}.tif"))


def test_remote_paths_keep_their_scheme(bucket: str) -> None:
    files = flat(
        build_raster(times=2).gs.to_cog(f"{bucket}/{{time:%Y%m%d}}/{{band}}.tif")
    )

    assert [str(f) for f in files] == [
        f"{bucket}/{day}/{band}.tif" for day in DAYS for band in ("red", "nir")
    ]


def test_layout_plans_one_item_per_date_with_a_key_per_file() -> None:
    items = layout.layout(build_raster(times=2), "out/{time:%Y%m%d}/{band}.tif")

    assert [list(scene.files) for scene in items] == [["red", "nir"], ["red", "nir"]]
    first = items[0].files["red"]
    assert str(first.path) == "out/20250601/red.tif"
    assert first.raster.gs.variables == ("red",)
    assert "time" not in first.raster.dims


def test_an_unsplit_file_is_the_image_asset() -> None:
    items = layout.layout(build_raster(times=2), "out/{time:%Y%m%d}.tif")

    assert [list(scene.files) for scene in items] == [["image"], ["image"]]
    assert items[0].files["image"].raster.gs.variables == ("red", "nir")


def test_an_undated_raster_is_one_item() -> None:
    (scene,) = layout.layout(build_raster(), "out/s_{band}.tif")

    assert list(scene.files) == ["red", "nir"]


def test_write_writes_exactly_the_layout(tmp_path: Path) -> None:
    source = build_raster(times=2)
    template = f"{tmp_path}/{{time:%Y%m%d}}/{{band}}.tif"

    written = source.gs.to_cog(template)

    assert [{k: str(f) for k, f in s.written.items()} for s in written] == [
        {k: str(f.path) for k, f in s.files.items()}
        for s in layout.layout(source, template)
    ]


def test_a_write_returns_one_item_per_date(tmp_path: Path) -> None:
    items = build_raster(times=2).gs.to_cog(f"{tmp_path}/{{time:%Y%m%d}}_{{band}}.tif")

    assert [{k: f.name for k, f in s.written.items()} for s in items] == [
        {"red": "20250601_red.tif", "nir": "20250601_nir.tif"},
        {"red": "20250602_red.tif", "nir": "20250602_nir.tif"},
    ]


def test_each_file_states_its_band_and_date(tmp_path: Path) -> None:
    scene = build_raster(times=2)

    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif"))

    headers = [open_raster(f) for f in files]
    assert [h.gs.variables for h in headers] == [("red",), ("nir",), ("red",), ("nir",)]
    assert len({str(f) for f in files}) == 4
    assert headers[0].gs.times[0] == pd.Timestamp(scene.time.values[0])
    assert headers[0].gs.acdd is None


def test_an_unsplit_file_lists_every_band_in_order(tmp_path: Path) -> None:
    (file,) = flat(build_raster().gs.to_cog(f"{tmp_path}/scene.tif"))

    header = open_raster(file)
    assert header.gs.variables == ("red", "nir")
    assert header.gs.times is None
    assert Path(file).is_absolute()


def test_a_tree_names_each_scene_by_its_collection(tmp_path: Path) -> None:
    from geosave.geodata import tree

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    sample = tree({"optical": image, "label": label})

    scenes = sample.gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    assert [scene.collection for scene in scenes] == ["optical", "label"]
    files = flat(scenes)
    assert files[0].name.startswith("optical_")
    assert files[1].name == "label.tif"


def test_a_scalar_time_fills_the_time_field(tmp_path: Path) -> None:
    # A scalar time coordinate is one date, as one step of a time axis is.
    scene = build_raster(times=2).isel(time=1)

    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))

    assert [file.name for file in files] == ["s_20250602.tif"]
    restored = open_raster(files[0])
    assert pd.Timestamp(restored.time.values) == pd.Timestamp("2025-06-02")


def test_to_cog_returns_each_file_with_the_header_it_states(scene, tmp_path) -> None:
    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif"))

    assert [file.name for file in files] == [
        "s_20250601_red.tif",
        "s_20250601_nir.tif",
        "s_20250611_red.tif",
        "s_20250611_nir.tif",
    ]
    assert [file.header.gs.variables for file in files][:2] == [("red",), ("nir",)]
    assert [list(file.header.gs.times)[0] for file in files][::2] == [
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-06-11"),
    ]
    assert Path(files[0]).exists()


def test_a_relative_template_returns_absolute_paths(scene, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (path, _) = flat(scene.gs.to_cog("out/s_{time:%Y%m%d}.tif"))

    assert Path(path).is_absolute()
    assert Path(path) == tmp_path / "out" / "s_20250601.tif"


def test_a_store_keeps_its_dates_in_one_file() -> None:
    (scene,) = layout.layout(build_raster(times=2), "out/scene.zarr")

    assert list(scene.files) == ["image"]
    assert scene.files["image"].raster.sizes["time"] == 2


def test_a_tree_lays_out_one_store_per_layer() -> None:
    from geosave.geodata import tree

    image = build_raster(times=2)
    sample = tree({"optical": image, "dem": image.isel(time=0, drop=True)[["red"]]})

    items = layout.layout(sample, "out/{collection}.nc")

    assert [{k: p.path.name for k, p in s.files.items()} for s in items] == [
        {"image": "optical.nc"},
        {"red": "dem.nc"},
    ]


def test_a_upath_template_carries_its_options_to_every_file() -> None:
    template = UPath("memory://bucket/{band}.tif", skip_instance_cache=True)

    (scene,) = layout.layout(build_raster(), template)

    assert {
        p.path.storage_options["skip_instance_cache"] for p in scene.files.values()
    } == {True}


def test_save_writes_each_planned_file_through_the_writer() -> None:
    plan = layout.layout(build_raster(times=2), "out/{time:%Y%m%d}.tif")
    seen = []

    def write(raster, path):
        seen.append(path.name)
        return RasterFile(path, raster)

    scenes = layout.save(plan, write)

    assert seen == ["20250601.tif", "20250602.tif"]
    assert [list(scene.written) for scene in scenes] == [["image"], ["image"]]


def test_save_defers_every_write_when_asked() -> None:
    plan = layout.layout(build_raster(times=2), "out/{time:%Y%m%d}.tif")
    seen = []

    def write(raster, path):
        seen.append(path.name)
        return RasterFile(path, raster)

    pending = layout.save(plan, write, compute=False)

    assert seen == []
    (scenes,) = dask.compute(pending)
    assert sorted(seen) == ["20250601.tif", "20250602.tif"]
    assert all(scene.pending is None for scene in scenes)
