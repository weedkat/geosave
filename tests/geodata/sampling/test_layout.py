"""Blocks group chips so each stored chunk is read once, or again only at its edges."""

from odc.geo.geobox import GeoBox

from geosave.geodata.sampling import (
    ChipLocation,
    ChipMode,
    ChipSource,
    find_block_ends,
    lay_chip_blocks,
    lay_chip_offsets,
    lay_chips_geobox,
)

SCENE = GeoBox.from_bbox(
    (500_000.0, 9_000_000.0, 510_240.0, 9_010_240.0),
    crs="EPSG:32749",
    shape=(1024, 1024),
    tight=True,
)


def lay_rows(size: int, overlap: int | float, mode: ChipMode = "reflect"):
    """Lay chip rows over the square scene, as the chipper does."""
    source = ChipSource(0, SCENE, None)
    offsets = lay_chip_offsets(SCENE.height, size=size, overlap=overlap, mode=mode)
    rows = []
    for top in offsets:
        row = []
        for left in offsets:
            row.append(ChipLocation(source, top, left))
        rows.append(row)
    return rows


def read_tops(block):
    tops = []
    for location in block.locations:
        if location.top not in tops:
            tops.append(location.top)
    return tops


def read_lefts(block):
    lefts = []
    for location in block.locations:
        if location.left not in lefts:
            lefts.append(location.left)
    return lefts


def test_blocks_end_at_chunk_ends_at_least_min_length_apart():
    chunk_lengths = (256, 256, 256, 256, 102)

    assert find_block_ends(chunk_lengths, 1126, min_length=896) == [1024, 1126]
    assert find_block_ends(chunk_lengths, 1126, min_length=1) == [
        256,
        512,
        768,
        1024,
        1126,
    ]
    # Pixels in memory are one block.
    assert find_block_ends(None, 1126, min_length=896) == [1126]


def test_chips_geobox_reaches_past_the_edge_only_as_far_as_the_last_chip():
    rows = lay_rows(224, 0.25)

    geobox = lay_chips_geobox(rows, 224)

    # Chip rows start at 0, 168, ..., 840; the last ends at 1064.
    assert geobox.shape == (1064, 1064)
    assert geobox.transform == SCENE.transform


def test_a_chip_row_straddling_a_chunk_row_is_read_with_the_next():
    rows = lay_rows(224, 0.25)
    geobox = lay_chips_geobox(rows, 224)

    blocks = lay_chip_blocks(rows, [256, 512, 768, 1064], [1064], 224)

    assert [read_tops(block) for block in blocks] == [
        [0],
        [168],
        [336, 504],
        [672, 840],
    ]
    assert [block.rows for block in blocks] == [
        slice(0, 256),
        slice(168, 512),
        slice(336, 768),
        slice(672, 1064),
    ]
    assert blocks[1].columns == slice(0, geobox.width)


def test_a_chip_straddling_a_chunk_column_is_read_with_the_next():
    rows = lay_rows(224, 0.25)

    blocks = lay_chip_blocks(rows, [512, 1064], [512, 1064], 224)

    assert [(read_tops(block), read_lefts(block)) for block in blocks] == [
        ([0, 168], [0, 168]),
        ([0, 168], [336, 504, 672, 840]),
        ([336, 504, 672, 840], [0, 168]),
        ([336, 504, 672, 840], [336, 504, 672, 840]),
    ]
    assert (blocks[1].rows, blocks[1].columns) == (slice(0, 512), slice(336, 1064))
    # A chip's pixels within its block.
    location = blocks[1].locations[1]
    assert blocks[1].find_chip_pixels(location, 224) == (slice(0, 224), slice(168, 392))


def test_chips_inside_one_chunk_share_a_block():
    rows = lay_rows(128, 0)

    blocks = lay_chip_blocks(rows, [512, 1024], [512, 1024], 128)

    assert len(blocks) == 4
    assert [(block.rows, block.columns) for block in blocks] == [
        (slice(0, 512), slice(0, 512)),
        (slice(0, 512), slice(512, 1024)),
        (slice(512, 1024), slice(0, 512)),
        (slice(512, 1024), slice(512, 1024)),
    ]
    assert [len(block.locations) for block in blocks] == [16] * 4


def test_pixels_in_memory_are_one_block_holding_every_chip():
    rows = lay_rows(224, 0.25)
    geobox = lay_chips_geobox(rows, 224)

    blocks = lay_chip_blocks(rows, [geobox.height], [geobox.width], 224)

    assert len(blocks) == 1
    assert (blocks[0].rows, blocks[0].columns) == (slice(0, 1064), slice(0, 1064))
    assert len(blocks[0].locations) == 36


def test_every_chip_is_in_exactly_one_block():
    rows = lay_rows(200, 0.5, mode="drop")

    blocks = lay_chip_blocks(rows, [256, 512, 768, 1024], [256, 512, 768, 1024], 200)

    locations = []
    for block in blocks:
        locations.extend(block.locations)
    expected = []
    for row in rows:
        expected.extend(row)
    assert (
        sorted(locations, key=lambda location: (location.top, location.left))
        == expected
    )
    assert len(set(locations)) == len(expected)
