"""Native model tensor conversion and numerical processing contracts."""

import numpy as np
import pytest
import torch

from geosave.model.processing import normalize, to_raster


def test_bulk_encoding_shares_reads_and_preserves_sample_and_band_order():
    from dask import delayed
    import dask.array as da
    from geosave.model.processing import encode_rasters
    from tests.model.spec.operations import scene

    reads = []

    @delayed
    def read_pixels():
        reads.append(1)
        return np.full((4, 4), 10, dtype="uint16")

    raster = scene()
    raster["red"].data = da.from_delayed(read_pixels(), shape=(4, 4), dtype="uint16")
    raster.red.attrs.update(scale_factor=0.1, add_offset=1.0)
    raster["nir"] = raster.red.copy()
    raster.nir.attrs.update(scale_factor=0.2, add_offset=2.0)
    other = raster.assign_coords(x=raster.x + 1000)
    smaller = raster.isel(x=slice(0, 2))

    images = encode_rasters(
        [other, raster, smaller], bands=["nir", "red"], dims=["band", "y", "x"]
    )

    assert reads == [1]
    assert isinstance(images, tuple)
    assert [tuple(image.shape) for image in images] == [(2, 4, 4), (2, 4, 4), (2, 4, 2)]
    for image in images:
        assert (image[0] == 4).all()
        assert (image[1] == 2).all()
    assert raster.red.attrs["scale_factor"] == 0.1
    assert isinstance(raster.red.data, da.Array)


def test_bulk_encoding_rejects_later_incompatible_sample_before_reading():
    from dask import delayed
    import dask.array as da
    from geosave.model.processing import encode_rasters
    from tests.model.spec.operations import scene

    @delayed
    def unread_pixels():
        raise AssertionError("Invalid batch metadata must fail before pixel reads")

    raster = scene()
    raster["red"].data = da.from_delayed(unread_pixels(), shape=(4, 4), dtype="float32")
    temporal = raster.expand_dims(time=[0, 1])
    with pytest.raises(ValueError, match="position 1.*red.*time"):
        encode_rasters([raster, temporal], bands=["red"], dims=["band", "y", "x"])


def test_bulk_restoration_preserves_each_grid_and_checks_correspondence():
    from geosave.model.processing import restore_rasters
    from tests.model.spec.operations import scene

    raster = scene()
    shifted = raster.assign_coords(x=raster.x + 1000)
    values = torch.stack([torch.ones((4, 4)), torch.full((4, 4), 2.0)])
    values.requires_grad_()
    restored = restore_rasters(
        values, references=[raster, shifted], dims=["y", "x"], name="prediction"
    )
    assert restored[0].gs.geobox == raster.gs.geobox
    assert restored[1].gs.geobox == shifted.gs.geobox
    assert (restored[0].values == 1).all()
    assert (restored[1].values == 2).all()
    assert values.requires_grad

    with pytest.raises(ValueError, match="2 predictions for 1 reference"):
        restore_rasters(values, references=[raster], dims=["y", "x"], name="prediction")
    with pytest.raises(ValueError, match="position 1.*shape differs"):
        restore_rasters(
            [values[0], torch.ones((2, 2))],
            references=[raster, shifted],
            dims=["y", "x"],
            name="prediction",
        )


def test_normalization_uses_the_declared_axis_and_preserves_gradients():
    image = torch.full((2, 3, 2, 4, 4), 5.0, requires_grad=True)
    result = normalize(image, mean=[1, 2], std=[2, 4], dim=-3)
    assert result.shape == image.shape
    torch.testing.assert_close(result[:, :, 0], torch.full((2, 3, 4, 4), 2.0))
    torch.testing.assert_close(result[:, :, 1], torch.full((2, 3, 4, 4), 0.75))
    result.sum().backward()
    assert (image.grad[:, :, 0] == 0.5).all()
    assert (image.grad[:, :, 1] == 0.25).all()
    assert (image == 5).all()


def test_normalization_preserves_missing_pixels_and_rejects_invalid_statistics():
    image = torch.tensor([[[float("nan"), float("inf"), 2.0]]])
    result = normalize(image, mean=[1], std=[2], dim=0)
    assert result.isnan()[0, 0, 0]
    assert result.isinf()[0, 0, 1]
    assert result[0, 0, 2] == 0.5
    with pytest.raises(ValueError, match="positive"):
        normalize(image, mean=[1], std=[0], dim=0)
    with pytest.raises(ValueError, match="match"):
        normalize(image, mean=[1, 2], std=[1, 2], dim=0)


def test_output_grid_mismatch_is_rejected_instead_of_relabelled():
    from tests.model.spec.operations import scene

    with pytest.raises(ValueError, match="reference grid"):
        to_raster(
            np.zeros((1, 8, 8)),
            reference=scene(),
            dims=["band", "y", "x"],
            name="prediction",
        )


def test_tensor_conversion_never_broadcasts_a_static_band_over_time():
    import xarray as xr
    from geosave.model.processing import to_tensor

    raster = xr.Dataset(
        {
            "red": (("time", "y", "x"), np.ones((2, 2, 2))),
            "nir": (("y", "x"), np.ones((2, 2))),
        }
    )
    with pytest.raises(ValueError, match="dimension"):
        to_tensor(raster, dims=["band", "time", "y", "x"])


def test_model_requirements_preserve_lazy_pixels_and_report_mismatches():
    from dask import delayed
    import dask.array as da

    from geosave.model.processing import select_variables, validate_raster
    from tests.model.spec.operations import scene

    @delayed
    def unread_pixels():
        raise AssertionError("Checking metadata must not read pixels")

    raster = scene()
    raster["red"].data = da.from_delayed(unread_pixels(), shape=(4, 4), dtype="float32")
    raster["red"].attrs["units"] = "1"
    raster["nir"] = raster.red + 1
    selected = select_variables(raster, names=["nir", "red"])
    assert list(selected.data_vars) == ["nir", "red"]
    assert selected.red.data is raster.red.data
    assert selected.gs.geobox == raster.gs.geobox
    assert (
        validate_raster(selected, dims=["y", "x"], sizes={"y": 4}, units={"red": "1"})
        is selected
    )

    with pytest.raises(ValueError, match="red.*dimensions.*time"):
        validate_raster(raster, dims=["time", "y", "x"])
    with pytest.raises(ValueError, match="time.*None.*expected 3"):
        validate_raster(raster, sizes={"time": 3})
    with pytest.raises(ValueError, match="red.*units.*kelvin"):
        validate_raster(raster, units={"red": "kelvin"})
    with pytest.raises(ValueError, match="missing variable 'blue'"):
        validate_raster(raster, units={"blue": "1"})
    with pytest.raises(ValueError, match="blue.*available.*red"):
        select_variables(raster, names=["blue"])
    with pytest.raises(ValueError, match="repeated"):
        select_variables(raster, names=["red", "red"])


def test_builtin_recipe_selects_decodes_and_normalizes_in_band_order(tmp_path):
    from geosave.model.spec import ModelSpec
    from tests.model.spec.operations import scene

    raster = scene(10)
    raster.red.attrs.update(scale_factor=0.1, add_offset=1.0)
    raster["nir"] = raster.red.copy()
    raster.nir.attrs.update(scale_factor=0.2, add_offset=2.0)
    spec = ModelSpec.from_yaml("""
schema_version: 1
entry: checked
output: normalized
inputs:
  raster: {}
steps:
  selected:
    call: geosave.model.processing.select_variables
    kwargs: {raster: !ref raster, names: [nir, red]}
  checked:
    call: geosave.model.processing.validate_raster
    kwargs: {raster: !ref selected, dims: [y, x], sizes: {y: 4, x: 4}}
  decoded:
    call: !ref entry.gs.mask_and_scale
  encoded:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref decoded, dims: [band, y, x]}
  normalized:
    call: geosave.model.processing.normalize
    kwargs: {value: !ref encoded, mean: [2, 1], std: [2, 4], dim: -3}
""")
    path = tmp_path / "model_spec.yaml"
    spec.save(path)
    from geosave.geodata.io import open_raster

    source = tmp_path / "scene.nc"
    raster.to_netcdf(source)
    with open_raster(source) as opened:
        result = ModelSpec.load(path).run(raster=opened)

    torch.testing.assert_close(result[0], torch.ones((4, 4)))
    torch.testing.assert_close(result[1], torch.full((4, 4), 0.25))
    assert raster.red.attrs["scale_factor"] == 0.1
    assert list(raster.data_vars) == ["red", "nir"]
