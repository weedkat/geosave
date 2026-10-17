from geosave.geodata.sampling.chip import create_tiler


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
