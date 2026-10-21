import pytest

from geosave.geodata.sampling.chip import create_tiler


def test_chip_resolution_builds_a_grid_without_opening_assets(items, grid):
    samples = items.gs.sample()
    samples["items"] = samples["items"].str.replace(".tif", ".missing", regex=False)

    chips = samples.gs.chip((50, 70), resolution=5)

    assert len(chips) == 4
    assert chips.iloc[0].gs.geobox == grid.zoom_out(0.5)[:50, :70]
    assert chips.iloc[-1].gs.geobox.resolution.x == 5


@pytest.mark.parametrize("resolution", [0, -10, float("nan"), float("inf")])
def test_chip_rejects_invalid_resolution(items, resolution):
    with pytest.raises(ValueError, match="resolution"):
        items.gs.sample().gs.chip(32, resolution=resolution)


def test_chip_rejects_resolution_with_an_explicit_grid(items, grid):
    with pytest.raises(ValueError, match="resolution"):
        items.gs.sample().gs.chip(32, resolution=5, target=grid)


@pytest.mark.parametrize("inherit", [False, True])
def test_chip_resolution_only_needs_source_crs_metadata(items, inherit):
    from geosave.geodata.stac import create_catalog

    entries = items.gs.to_items()
    for entry in entries:
        for asset in entry.assets.values():
            asset.href = "/does-not-exist.tif"
            asset.extra_fields.pop("proj:shape", None)
            asset.extra_fields.pop("proj:transform", None)
            if inherit:
                entry.properties["proj:code"] = asset.extra_fields.pop("proj:code")

    chips = create_catalog(entries).gs.sample().gs.chip((25, 35), resolution=10)

    assert len(chips) == 4
    assert chips.iloc[0].gs.geobox.crs.epsg == 32633


def test_chip_resolution_requires_a_stated_crs(items):
    from geosave.geodata.stac import create_catalog

    entries = items.gs.to_items()
    for entry in entries:
        entry.properties.pop("proj:code", None)
        entry.properties.pop("proj:wkt2", None)
        for asset in entry.assets.values():
            asset.extra_fields.pop("proj:code", None)
            asset.extra_fields.pop("proj:wkt2", None)
    samples = create_catalog(entries).gs.sample()

    with pytest.raises(ValueError, match="CRS"):
        samples.gs.chip(32, resolution=10)


def test_fractional_overlap_scales_each_chip_dimension(items, grid):
    chips = items.gs.sample().gs.chip((20, 40), overlap=0.25, target=grid)

    # 5 shared rows and 10 shared columns give steps of 15 and 30 pixels.
    assert len(chips) == 12
    assert chips.iloc[0]["pad_widths"] == [[7, 0], [15, 0]]
    first = chips.iloc[0].gs.geobox
    second = chips.iloc[1].gs.geobox
    next_row = chips.iloc[3].gs.geobox
    assert second.transform.c - first.transform.c == 300
    assert next_row.transform.f - first.transform.f == -150


def test_chip_resolution_uses_each_frames_crs_without_opening_assets(items, grid):
    from odc.geo.geobox import GeoBox

    from geosave.geodata.stac import create_catalog
    from geosave.geodata.stac.extensions import projection

    optical = items[items.collection == "s2"].gs.to_items()
    shifted = optical[0].clone()
    shifted.id = "other-zone"
    other = GeoBox(grid.shape, grid.transform, "EPSG:32634")
    shifted.geometry = other.extent.to_crs("EPSG:4326").json
    shifted.bbox = list(other.extent.to_crs("EPSG:4326").boundingbox)
    for asset in shifted.assets.values():
        projection.write_asset(asset, other)
    entries = [*optical, shifted]
    for entry in entries:
        for asset in entry.assets.values():
            asset.href = "/does-not-exist.tif"
    samples = create_catalog(entries).gs.sample()
    frames = samples.gs.frame(windows={"s2": {"slots": ["0D"]}})

    chips = frames.gs.chip((25, 35), resolution=10)

    assert len(chips) == 24
    assert chips.iloc[0].gs.geobox == grid[:25, :35]
    assert chips.iloc[-1].gs.geobox == other[25:, 35:]
    assert chips["proj:code"].value_counts().to_dict() == {
        "EPSG:32633": 20,
        "EPSG:32634": 4,
    }


def test_chip_lays_chips_over_the_sample_grid(items, grid):
    samples = items.gs.sample()

    chips = samples.gs.chip((25, 35), target=grid)

    assert len(chips) == 4
    assert chips["proj:shape"].tolist() == [[25, 35]] * 4
    assert all(widths == [[0, 0], [0, 0]] for widths in chips["pad_widths"])
    assert chips["pad_mode"].isna().all()
    assert chips.iloc[3].gs.anchor.geobox == grid[25:50, 35:70]


def test_overlapping_chips_pad_past_the_region(items, grid):
    chips = items.gs.sample().gs.chip(32, overlap=8, pad_mode="edge", target=grid)

    first = chips.iloc[0]
    assert first["pad_widths"] == [[12, 0], [12, 0]]
    assert first["pad_mode"] == "edge"
    assert first.gs.geobox.enclosing(first.gs.region).shape == (20, 20)


def test_chip_footprints_are_their_grids_in_wgs84(items, grid):
    chip = items.gs.sample().gs.chip(25, target=grid).iloc[0]

    expected = chip.gs.anchor.geobox.extent.to_crs("EPSG:4326").geom
    assert chip.geometry.equals_exact(expected, 1e-9)


def test_chipping_a_chip_cuts_its_read_region(items, grid):
    chips = items.gs.sample().gs.chip(32, overlap=8, target=grid)

    nested = chips.iloc[[0]].gs.chip(10, target=grid)

    assert (
        nested.iloc[0].gs.anchor.geobox
        == chips.iloc[0].gs.geobox.enclosing(chips.iloc[0].gs.region)[0:10, 0:10]
    )


def test_create_tiler_adds_a_halo_only_for_overlap():
    _, flush_halo = create_tiler((300, 500), (128, 128), overlap=0, pad_mode="reflect")
    _, halo = create_tiler((300, 500), (128, 128), overlap=32, pad_mode="reflect")

    assert flush_halo == [(0, 0), (0, 0)]
    assert halo == [(48, 48), (48, 48)]


def test_chip_uses_target_pixels_without_opening_assets(items, grid):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    samples = items.gs.sample()
    # The same footprint is 100 x 140 pixels at 5 m, rather than 50 x 70.
    target = GeoBox((100, 140), grid.transform * Affine.scale(0.5), grid.crs)
    samples["items"] = samples["items"].str.replace(".tif", ".missing", regex=False)

    chips = samples.gs.chip((50, 70), target=target)

    assert len(chips) == 4
    assert chips.iloc[0].gs.geobox == target[:50, :70]
    assert chips.iloc[3].gs.geobox == target[50:, 70:]
    assert chips.iloc[0]["parent"] == samples.iloc[0].gs.key


def test_chip_target_supplies_lattice_without_restricting_sample_extent(items, grid):
    # A one-pixel template can define the lattice over a larger sample.
    chips = items.gs.sample().gs.chip((25, 35), target=grid[:1, :1])

    assert len(chips) == 4
    assert chips.iloc[3].gs.geobox == grid[25:, 35:]


def test_chip_covers_sample_on_shifted_target_lattice(items, grid):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    target = GeoBox((1, 1), Affine(5, 0, 300002.5, 0, -5, 5000497.5), grid.crs)
    chips = items.gs.sample().gs.chip((101, 141), target=target)

    assert len(chips) == 1
    expected = GeoBox((101, 141), Affine(5, 0, 299997.5, 0, -5, 5000502.5), grid.crs)
    assert chips.iloc[0].gs.geobox == expected


def test_chip_target_can_use_a_different_crs(items):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    samples = items.gs.sample()
    target = GeoBox((1, 1), Affine(20, 0, 0, 0, -20, 0), "EPSG:3857")
    chips = samples.gs.chip(32, target=target)

    assert len(chips) > 0
    row = chips.iloc[0]
    assert str(row.gs.geobox.crs) == "EPSG:3857"
    assert row.gs.geobox.resolution.xy == (20, -20)
    region = row.gs.parent_grid
    assert region.extent.contains(samples.iloc[0].gs.region.to_crs("EPSG:3857"))


def test_chip_resolution_is_independent_of_asset_order(items):
    from geosave.geodata.stac import create_catalog

    entries = items[items.collection == "s2"].gs.to_items()
    entry = entries[0]
    first = next(iter(entry.assets.values()))
    coarse = first.clone()
    coarse.extra_fields["proj:shape"] = [25, 35]
    coarse.extra_fields["proj:transform"] = [20, 0, 300005, 0, -20, 5000505]
    entry.add_asset("coarse", coarse)
    before = create_catalog([entry]).gs.sample().gs.chip(25, resolution=10)
    entry.assets = dict(reversed(list(entry.assets.items())))

    after = create_catalog([entry]).gs.sample().gs.chip(25, resolution=10)

    assert before["proj:transform"].tolist() == after["proj:transform"].tolist()
    assert before.iloc[0].gs.geobox.resolution.x == 10


def test_chip_resolution_rejects_mixed_crs_within_one_row(items, grid):
    from odc.geo.geobox import GeoBox

    from geosave.geodata.stac import create_catalog
    from geosave.geodata.stac.extensions import projection

    entries = items[items.collection == "s2"].gs.to_items()
    for asset in entries[0].assets.values():
        projection.write_asset(asset, GeoBox(grid.shape, grid.transform, "EPSG:32634"))
    samples = create_catalog(entries).gs.sample()

    with pytest.raises(ValueError, match="CRS"):
        samples.gs.chip(32, resolution=10)

    assert samples.gs.chip(32, target=grid).iloc[0].gs.geobox.crs == grid.crs


def test_chip_persists_alignment_kernels_and_reads_lazily(tmp_path):
    import dask.array as da
    import numpy as np
    from odc.geo.geobox import GeoBox

    import geosave as gs

    grid = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:3857", resolution=20)
    entries = []
    for collection, name, values, nodata in (
        ("image", "value", [[0.0, 10.0], [20.0, 30.0]], -9999),
        ("label", "label", [[1, 2], [3, 4]], 255),
    ):
        values = np.asarray(
            values, dtype="float32" if collection == "image" else "uint8"
        )
        source = gs.raster({name: (("y", "x"), values)}, grid, nodata=nodata)
        path = tmp_path / f"{collection}.tif"
        source.gs.to_cog(path)
        entries.extend(gs.stac.create_items({name: path}, collection=collection))
    samples = gs.stac.create_catalog(entries).gs.sample()
    chips = samples.gs.chip(
        4,
        resolution=10,
        resampling={"image/value": "bilinear", "label/label": "nearest"},
    )
    restored = gs.open_vector(chips.gs.to_geoparquet(tmp_path / "chips.parquet"))

    tree = restored.iloc[0].gs.to_xarray()

    image = tree["image"].dataset.value
    label = tree["label"].dataset.label
    assert isinstance(image.data, da.Array)
    assert isinstance(label.data, da.Array)
    assert label.dtype == "uint8"
    assert tree["image"].dataset.gs.geobox == restored.iloc[0].gs.geobox
    assert tree["label"].dataset.gs.geobox == restored.iloc[0].gs.geobox
    np.testing.assert_allclose(
        image.compute(),
        [
            [0, 2.5, 7.5, 10],
            [5, 7.5, 12.5, 15],
            [15, 17.5, 22.5, 25],
            [20, 22.5, 27.5, 30],
        ],
    )
    np.testing.assert_array_equal(
        label.compute(),
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
    )


def test_gridless_sample_requires_explicit_chip_grid(items, grid):
    samples = items.gs.sample()
    with pytest.raises(ValueError, match="target"):
        samples.gs.chip(32)
    chips = samples.gs.chip(32, target=grid)
    assert all(row.gs.geobox is not None for _, row in chips.iterrows())
