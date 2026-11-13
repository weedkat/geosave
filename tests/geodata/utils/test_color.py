import pytest

from geosave.geodata.utils.color import parse_color


@pytest.mark.parametrize(
    ("color", "rgb"),
    [
        ("#ff8000", (255, 128, 0)),
        ("FF8000", (255, 128, 0)),
        ((255, 128, 0), (255, 128, 0)),
        ([255, 128, 0], (255, 128, 0)),
    ],
)
def test_parse_color_reads_hex_and_triples_as_an_rgb_tuple(color, rgb) -> None:
    assert parse_color(color) == rgb
