"""What `geodata.io` offers callers."""


def test_cogfile_is_not_public() -> None:
    from geosave.geodata import io
    from geosave.geodata.io import raster

    assert "CogFile" not in io.__all__
    assert "CogFile" not in raster.__all__
