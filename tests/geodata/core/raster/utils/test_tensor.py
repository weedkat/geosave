import pytest


from tests.geodata.core.raster.conftest import optical

UTM = "EPSG:32633"


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_to_tensor_casts_to_a_named_or_torch_dtype(torch, dtype, expected) -> None:
    dtype = dtype if dtype == "float32" else getattr(torch, dtype)
    expected = getattr(torch, expected)
    assert optical().gs.to_tensor(dtype=dtype).dtype is expected
    assert optical().B04.gs.to_tensor(dtype=dtype).dtype is expected


def test_pixels_convert_to_one_tensor(torch):
    for raster in [optical(), optical().B04]:
        assert isinstance(raster.gs.to_tensor(), torch.Tensor)
