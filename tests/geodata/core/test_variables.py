import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core.variables import variable_names


def scene() -> xr.Dataset:
    band = (("y", "x"), np.zeros((2, 2)))
    return xr.Dataset({"red": band, "green": band, "nir": band}, coords={"y": [0, 1]})


def test_no_target_names_every_data_variable_in_order() -> None:
    assert variable_names(scene(), None) == ["red", "green", "nir"]


def test_one_name_is_a_target_of_its_own() -> None:
    assert variable_names(scene(), "nir") == ["nir"]


def test_several_names_keep_the_order_given() -> None:
    assert variable_names(scene(), ("nir", "red")) == ["nir", "red"]


@pytest.mark.parametrize("target", ["blue", ["red", "blue"], ["y"]])
def test_a_name_that_is_not_a_data_variable_is_refused(target) -> None:
    with pytest.raises(KeyError, match="carries \\['red', 'green', 'nir'\\]"):
        variable_names(scene(), target)
