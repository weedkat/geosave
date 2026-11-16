"""The dense preprocessor prepares and standardizes stored bands and registers by role."""

import numpy as np
import pytest
import xarray as xr

import geosave  # noqa: F401  registers the .gs accessors
from geosave.ml.factory import BuildSpec
from geosave.ml.processing.raster.preprocessing.dense import DensePreprocessor
from geosave.ml.processing.raster.preprocessing.base import (
    PREPROCESSORS,
    RasterPreprocessor,
)


@pytest.fixture(autouse=True)
def restore_preprocessors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PREPROCESSORS, "_factories", dict(PREPROCESSORS))


def test_preprocess_decodes_selected_bands_and_fills_missing_pixels():
    stored = xr.DataArray(
        np.array([[1000, 0]], dtype="uint16"),
        dims=("y", "x"),
        attrs={"scale_factor": 1e-4, "add_offset": 0.0, "_FillValue": 0},
    )
    raster = xr.Dataset({"B04": stored, "B08": stored * 2})
    rasters = xr.DataTree.from_dict({"image": raster})

    sample = DensePreprocessor(bands=["B08"])(rasters)

    assert list(sample.children) == ["image"]
    image = sample["image"].to_dataset()
    assert list(image.data_vars) == ["B08"]
    assert image["B08"].dtype == "float32"
    np.testing.assert_allclose(image["B08"].values, [[0.2, 0.0]])


def test_stored_values_are_taken_as_they_are_without_decoding():
    stored = xr.DataArray(
        np.array([[1000, 0]], dtype="uint16"),
        dims=("y", "x"),
        attrs={"scale_factor": 1e-4, "add_offset": 0.0, "_FillValue": 0},
    )
    rasters = xr.DataTree.from_dict({"image": xr.Dataset({"B08": stored})})

    sample = DensePreprocessor(bands=["B08"], decode=False)(rasters)

    assert sample["image"]["B08"].dtype == "float32"
    assert sample["image"]["B08"].values.tolist() == [[1000.0, 0.0]]


def test_missing_pixels_stay_missing_without_a_fill():
    stored = xr.DataArray(
        np.array([[1000, 0]], dtype="uint16"), dims=("y", "x"), attrs={"_FillValue": 0}
    )
    rasters = xr.DataTree.from_dict({"image": xr.Dataset({"B08": stored})})

    sample = DensePreprocessor(bands=["B08"], fill=None)(rasters)

    assert np.isnan(sample["image"]["B08"].values).tolist() == [[False, True]]


def test_the_raster_is_read_and_returned_under_the_configured_name():
    rasters = xr.DataTree.from_dict({"s2": xr.Dataset({"B08": ("x", [1.0])})})

    sample = DensePreprocessor(bands=["B08"], name="s2")(rasters)

    assert list(sample.children) == ["s2"]
    with pytest.raises(KeyError, match="image"):
        DensePreprocessor(bands=["B08"])(rasters)


def test_preprocess_standardizes_each_band_lazily():
    raster = xr.Dataset(
        {"B04": (("y", "x"), np.ones((1, 2))), "B08": (("y", "x"), np.ones((1, 2)))}
    ).chunk()
    rasters = xr.DataTree.from_dict({"image": raster})
    preprocessor = DensePreprocessor(
        ["B04", "B08"], decode=False, mean=[1.0, 0.0], std=[1.0, 2.0]
    )

    image = preprocessor(rasters)["image"].to_dataset()

    assert image["B04"].chunks is not None
    assert image["B04"].dtype == "float32"
    assert image["B04"].values.tolist() == [[0.0, 0.0]]
    assert image["B08"].values.tolist() == [[0.5, 0.5]]


def test_mean_without_std_is_refused():
    with pytest.raises(ValueError, match="both mean and std"):
        DensePreprocessor(["B04"], mean=[0.1])


def test_the_library_preprocessor_is_found_without_importing_its_module():
    assert PREPROCESSORS.resolve(BuildSpec(name="dense")) is DensePreprocessor


def test_a_custom_preprocessor_inherits_and_rebuilds_from_its_spec():
    @PREPROCESSORS.register("cloud_free")
    class CloudFree(DensePreprocessor):
        def __call__(self, rasters):
            image = super().__call__(rasters)[self.name].to_dataset()
            clear = rasters["cloud"].to_dataset()["mask"] == 0
            return xr.DataTree.from_dict({self.name: image.where(clear, 0)})

    spec = PREPROCESSORS.get_spec(
        BuildSpec(name="cloud_free", init_args={"bands": ["red"]})
    )
    rebuilt = PREPROCESSORS.build(BuildSpec.model_validate(spec.model_dump()))
    red = xr.DataArray(np.array([[4.0, 6.0]]), dims=("y", "x"))
    mask = xr.DataArray(np.array([[0, 1]]), dims=("y", "x"))
    rasters = xr.DataTree.from_dict(
        {"image": xr.Dataset({"red": red}), "cloud": xr.Dataset({"mask": mask})}
    )

    sample = rebuilt(rasters)

    assert spec.init_args == {
        "bands": ["red"],
        "decode": True,
        "fill": 0.0,
        "mean": None,
        "std": None,
        "name": "image",
    }
    assert sample["image"]["red"].values.tolist() == [[4.0, 0.0]]


def test_a_class_outside_the_role_is_refused():
    with pytest.raises(TypeError, match="Half is not a RasterPreprocessor"):

        @PREPROCESSORS.register("half")
        class Half:
            def __call__(self, rasters):
                return rasters


def test_a_role_without_its_method_cannot_be_built():
    class Nothing(RasterPreprocessor):
        pass

    with pytest.raises(TypeError, match="__call__"):
        Nothing()
