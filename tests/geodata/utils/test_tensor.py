import numpy as np
import pytest

from tests.geodata.core.raster.conftest import optical


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("float32", "float32"),
        ("int16", "int16"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_to_tensor_casts_to_a_named_or_torch_dtype(torch, dtype, expected) -> None:
    dtype = dtype if dtype == "float32" else getattr(torch, dtype)
    expected = getattr(torch, expected)
    assert optical().gs.to_tensor(dtype=dtype).dtype is expected
    assert optical().B04.gs.to_tensor(dtype=dtype).dtype is expected


def test_to_tensor_refuses_an_unknown_dtype_name(torch) -> None:
    with pytest.raises(ValueError, match="not-a-dtype"):
        optical().gs.to_tensor(dtype="not-a-dtype")


def test_to_tensor_refuses_a_dtype_that_is_not_torch_before_reading(torch) -> None:
    import dask

    lazy = optical().chunk({"y": 4})

    def fail(*_):
        raise AssertionError("pixels were read")

    with dask.callbacks.Callback(pretask=fail):
        with pytest.raises(TypeError, match="torch dtype"):
            lazy.gs.to_tensor(dtype=np.float32)
