"""A raster pipeline runs its sub-processors in order, predicts by chips, and reloads from a release."""

import json

import geopandas as gpd
import numpy as np
import pytest
import shapely
import torch
import xarray as xr
from odc.geo.geobox import GeoBox
from transformers import AutoModel

import geosave  # noqa: F401  registers the .gs accessors
from geosave.geodata.io import open_raster
from geosave.ml import ChangePipeline, ModelChain, build_model
from geosave.ml.processing.raster.ingestion.base import RasterIngestor
from geosave.ml.processing.raster.pipeline import RasterPipeline
from geosave.ml.processing.raster.postprocessing.base import RasterPostprocessor
from geosave.ml.processing.raster.postprocessing.dense import (
    NO_PREDICTION,
    RegressionPostprocessor,
    SegmentationPostprocessor,
    copy_scalar_coords,
)
from geosave.ml.processing.raster.prediction.base import RasterPredictor
from geosave.ml.processing.raster.prediction.dense import DensePredictor
from geosave.ml.processing.raster.preprocessing.base import (
    PREPROCESSORS,
    RasterPreprocessor,
)
from geosave.ml.processing.raster.preprocessing.dense import DensePreprocessor
from geosave.ml.processing.change.preprocessing.base import ChangePreprocessor
from geosave.ml.release import ChainModel, load_pipeline
from tests.geodata.sampling.conftest import build_scene

BANDS = ["red", "nir"]


def as_rasters(raster: xr.Dataset) -> xr.DataTree:
    return xr.DataTree.from_dict({"image": raster})


class PixelModel(torch.nn.Module):
    """Scores each pixel alone, so merged chips must equal the whole raster."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.score(image)


class TwoScenes(RasterIngestor):
    def __call__(self, region):
        rasters = xr.DataTree.from_dict({"image": build_scene(times=None)})
        return [rasters, rasters]


class Halve(RasterPreprocessor):
    def __init__(self, bands):
        self.bands = bands

    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        image = rasters["image"].to_dataset()[self.bands].astype("float32")
        sample = xr.DataTree.from_dict({"image": image / 2})
        return sample


class Strongest(RasterPostprocessor[xr.DataArray, xr.Dataset]):
    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset:
        class_map = output.argmax("band").astype("uint8")
        product = copy_scalar_coords(class_map.to_dataset(name="class_id"), sample)
        return product


def build_pipeline(
    model: torch.nn.Module, context: list[str] | None = None
) -> RasterPipeline:
    return RasterPipeline(
        model,
        preprocessor=Halve(BANDS),
        predictor=DensePredictor(size=4, stride=3, context=context),
        postprocessor=Strongest(),
        ingestor=TwoScenes(),
    )


def test_sub_processors_are_called_directly():
    pipeline = build_pipeline(PixelModel().eval())
    rasters = pipeline.ingestor(None)[0]

    sample = pipeline.preprocessor(rasters)
    output = pipeline.predictor(pipeline.model, sample)
    product = pipeline.postprocessor(output, sample)

    assert output.dims == ("band", "y", "x")
    xr.testing.assert_identical(product, pipeline.predict(rasters))


def test_merged_chips_predict_what_the_whole_raster_would():
    model = PixelModel().eval()
    pipeline = build_pipeline(model)
    raster = build_scene(times=None)

    product = pipeline.predict(as_rasters(raster))

    pixels = np.stack([raster["red"].values, raster["nir"].values])
    with torch.inference_mode():
        whole = model(image=torch.from_numpy(pixels)[None] / 2)[0]
    assert product["class_id"].values.tolist() == whole.argmax(0).tolist()
    assert product.gs.geobox == raster.gs.geobox


def test_a_product_keeps_its_rasters_acquisition_time():
    pipeline = build_pipeline(PixelModel().eval())
    raster = build_scene().isel(time=2)

    product = pipeline.predict(as_rasters(raster))

    assert product["time"].values == raster["time"].values


def test_a_pipeline_carries_its_ingestor_without_running_it():
    pipeline = build_pipeline(PixelModel().eval())

    acquisitions = pipeline.ingestor(None)
    product = pipeline.predict(acquisitions[0])

    assert len(acquisitions) == 2
    assert product["class_id"].dtype == "uint8"


def test_a_pipeline_without_an_ingestor_predicts_rasters():
    pipeline = RasterPipeline(
        PixelModel().eval(), Halve(BANDS), DensePredictor(size=4), Strongest()
    )

    product = pipeline.predict(as_rasters(build_scene(times=None)))

    assert pipeline.ingestor is None
    assert product["class_id"].dtype == "uint8"


class PlaceModel(PixelModel):
    """Takes the chip's centre beside its pixels, as Clay and Prithvi-TL do."""

    def __init__(self) -> None:
        super().__init__()
        self.centroids: list[torch.Size] = []

    def forward(self, image: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        self.centroids.append(centroid.shape)
        return self.score(image)


def test_the_predictors_context_reaches_the_model_by_name():
    model = PlaceModel().eval()
    pipeline = build_pipeline(model, context=["centroid"])

    pipeline.predict(as_rasters(build_scene(times=None)))

    assert model.centroids[0] == torch.Size([8, 2])


class FusionModel(torch.nn.Module):
    """Takes two pixel inputs under its own names."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(2, 3, kernel_size=1)
        self.radar_means: list[float] = []

    def forward(self, optical: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        self.radar_means.append(float(radar.mean()))
        return self.score(torch.cat([optical, radar], dim=1))


def test_the_preprocessor_names_the_models_pixel_inputs():
    class Fused(RasterPreprocessor):
        def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
            optical = rasters["s2"].to_dataset()[["red"]].astype("float32")
            radar = rasters["s1"].to_dataset()[["vv"]].astype("float32")
            sample = xr.DataTree.from_dict({"optical": optical, "radar": radar * 0})
            return sample

    scene = build_scene(times=None)
    radar = scene[["nir"]].rename({"nir": "vv"})
    rasters = xr.DataTree.from_dict({"s2": scene, "s1": radar})
    model = FusionModel().eval()
    pipeline = RasterPipeline(
        model,
        Fused(),
        DensePredictor(size=4, stride=3),
        RegressionPostprocessor(names=["a", "b", "c"]),
    )

    product = pipeline.predict(rasters)

    assert product.gs.geobox == scene.gs.geobox
    # Each node reached the model under its own name, as the preprocessor left it.
    assert model.radar_means == [0.0, 0.0]


class CloudMasked(RasterPreprocessor):
    """Passes a mask beside the model's input, as a cloud-masking model would."""

    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        image = rasters["s2"].to_dataset()[BANDS].astype("float32")
        clear = (rasters["scl"]["scl"] != 9).to_dataset(name="clear")
        sample = xr.DataTree.from_dict({"image": image, "clear": clear})
        return sample


def build_masked_pipeline() -> RasterPipeline:
    return RasterPipeline(
        PixelModel().eval(),
        CloudMasked(),
        DensePredictor(size=4, stride=3, nodes=["image"]),
        SegmentationPostprocessor(
            ["a", "b", "c"], ["#000000", "#111111", "#222222"], mask="clear"
        ),
    )


def build_cloudy_rasters() -> xr.DataTree:
    scene = build_scene(times=None)
    scl = xr.zeros_like(scene["red"]).astype("uint8").rename("scl")
    scl[:3, :] = 9
    return xr.DataTree.from_dict({"s2": scene, "scl": scl.to_dataset()})


def test_the_postprocessor_masks_the_product_with_the_samples_mask():
    torch.manual_seed(0)
    rasters = build_cloudy_rasters()
    pipeline = build_masked_pipeline()

    sample = pipeline.preprocessor(rasters)
    output = pipeline.predictor(pipeline.model, sample)
    product = pipeline.predict(rasters)

    # The predictor hands over the model's values as they are, unmasked.
    assert np.isfinite(output.values).all()
    class_id = product["class_id"]
    assert class_id.dtype == "uint8"
    assert (class_id.values[:3] == NO_PREDICTION).all()
    assert (class_id.values[3:] != NO_PREDICTION).all()
    assert class_id.gs.nodata == NO_PREDICTION
    assert class_id.gs.flags is not None


def test_a_mask_is_applied_chunk_by_chunk_in_a_store(tmp_path):
    torch.manual_seed(0)
    pipeline = build_masked_pipeline()
    rasters = build_cloudy_rasters()
    expected = pipeline.predict(rasters)

    store = pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr", chunks=6)

    xr.testing.assert_identical(open_raster(store).compute(), expected)


class Footprint(RasterPostprocessor[xr.DataArray, gpd.GeoDataFrame]):
    """A task whose product is shapes: one row, the predicted raster's footprint."""

    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> gpd.GeoDataFrame:
        grid = output.gs.geobox
        product = gpd.GeoDataFrame(
            {"bands": [output.sizes["band"]]},
            geometry=[grid.extent.geom],
            crs=grid.crs.to_wkt(),
        )
        return product


def test_a_postprocessor_can_return_shapes():
    raster = build_scene(times=None)
    pipeline = RasterPipeline(
        PixelModel().eval(), Halve(BANDS), DensePredictor(size=4, stride=3), Footprint()
    )

    product = pipeline.predict(as_rasters(raster))

    assert product["bands"].tolist() == [3]
    assert shapely.equals(product.geometry.iloc[0], raster.gs.geobox.extent.geom)


def test_a_product_that_is_not_a_raster_cannot_be_written_to_a_store(tmp_path):
    pipeline = RasterPipeline(
        PixelModel().eval(), Halve(BANDS), DensePredictor(size=4, stride=3), Footprint()
    )
    rasters = as_rasters(build_scene(times=None))

    with pytest.raises(TypeError, match="writes raster products"):
        pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr")  # pyright: ignore[reportAttributeAccessIssue]


class WholeRaster(RasterPredictor[xr.DataArray]):
    """Predicts a sample in one model call, cutting no chips."""

    def __call__(self, model: torch.nn.Module, sample: xr.DataTree) -> xr.DataArray:
        image = sample["image"].to_dataset()
        pixels = torch.from_numpy(image.to_dataarray("band").values)[None]
        with torch.inference_mode():
            scores = model(image=pixels)[0].numpy()
        output = xr.DataArray(
            scores, dims=("band", "y", "x"), coords={"y": image.y, "x": image.x}
        )
        return output


def test_a_predictor_that_cuts_no_chips_predicts_in_memory_only(tmp_path):
    pipeline = RasterPipeline(
        PixelModel().eval(), Halve(BANDS), WholeRaster(), Strongest()
    )
    rasters = as_rasters(build_scene(times=None))

    product = pipeline.predict(rasters)

    assert product["class_id"].dtype == "uint8"
    with pytest.raises(NotImplementedError, match="WholeRaster states no chip size"):
        pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr")


class NeighbourModel(torch.nn.Module):
    """Scores each pixel from its neighbours, so a chunk seam would show."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.score(image)


@pytest.mark.parametrize("chunks", [3, 6, 7, 12, 64])
@pytest.mark.parametrize("stride", [2, 3, 4])
def test_a_store_holds_what_predicting_in_memory_returns(tmp_path, chunks, stride):
    torch.manual_seed(0)
    pipeline = RasterPipeline(
        NeighbourModel().eval(),
        DensePreprocessor(bands=BANDS, decode=False),
        DensePredictor(size=4, stride=stride, batch_size=1),
        RegressionPostprocessor(names=["a", "b", "c"]),
    )
    grid = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_230.0, 9_000_170.0),
        crs="EPSG:32749",
        shape=(17, 23),
        tight=True,
    )
    rasters = as_rasters(build_scene(grid).isel(time=2))
    expected = pipeline.predict(rasters)

    store = pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr", chunks=chunks)

    product = open_raster(store)
    xr.testing.assert_equal(product.compute(), expected)
    assert product.gs.geobox == expected.gs.geobox
    assert product["time"].values == expected["time"].values


def test_a_store_keeps_the_class_names_and_nodata_of_the_product(tmp_path):
    pipeline = RasterPipeline(
        PixelModel().eval(),
        DensePreprocessor(bands=BANDS, decode=False),
        DensePredictor(size=4, stride=3),
        SegmentationPostprocessor(["a", "b", "c"], ["#000000", "#111111", "#222222"]),
    )
    rasters = as_rasters(build_scene(times=None))
    expected = pipeline.predict(rasters)

    store = pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr", chunks=6)

    product = open_raster(store)
    xr.testing.assert_identical(product.compute(), expected)
    assert product["class_id"].gs.flags == expected["class_id"].gs.flags
    assert product["class_id"].gs.nodata == expected["class_id"].gs.nodata


def test_predicting_to_a_store_reads_one_chunk_with_its_halo_at_a_time(tmp_path):
    shapes = []

    class Recorded(DensePredictor):
        def __call__(self, model: torch.nn.Module, sample: xr.DataTree) -> xr.DataArray:
            shapes.append(tuple(sample.gs.common_geobox().shape))
            output = super().__call__(model, sample)
            return output

    pipeline = RasterPipeline(
        PixelModel().eval(), Halve(BANDS), Recorded(size=4, stride=3), Strongest()
    )
    rasters = as_rasters(build_scene(times=None).chunk(5))

    store = pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr", chunks=6)

    # Chips of 4 px every 3 px over 10 px: the first chunk (0-6) reads to the
    # end of the chip starting at 3; the second (6-10) reads back to that chip.
    assert shapes == [(7, 7), (7, 7), (7, 7), (7, 7)]
    assert open_raster(store)["class_id"].data.chunksize == (6, 6)


def test_predicting_to_a_store_refuses_an_existing_destination(tmp_path):
    pipeline = build_pipeline(PixelModel().eval())
    rasters = as_rasters(build_scene(times=None))
    pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr")

    with pytest.raises(FileExistsError):
        pipeline.predict_to_zarr(rasters, tmp_path / "product.zarr")


def _segmentation_chain():
    return build_model(
        {
            "encoder": {
                "name": "small_cnn",
                "init_args": {"in_channels": 2, "input_size": 16},
            },
            "decoder": {"name": "unet"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    ).eval()


def build_library_pipeline() -> RasterPipeline:
    return RasterPipeline(
        _segmentation_chain(),
        preprocessor=DensePreprocessor(bands=BANDS),
        predictor=DensePredictor(size=16, stride=12),
        postprocessor=SegmentationPostprocessor(
            ["a", "b", "c"], ["#000000", "#111111", "#222222"]
        ),
    )


def test_a_release_states_each_sub_processor_and_every_argument(tmp_path):
    pipeline = build_library_pipeline()

    ChainModel.from_pipeline(pipeline).save_pretrained(tmp_path / "release")

    config = json.loads((tmp_path / "release" / "config.json").read_text())
    assert config["preprocessor"] == {
        "name": "dense",
        "init_args": {
            "bands": BANDS,
            "decode": True,
            "fill": 0.0,
            "mean": None,
            "std": None,
            "name": "image",
        },
    }
    assert config["predictor"] == {
        "name": "dense",
        "init_args": {
            "size": 16,
            "stride": 12,
            "mode": "reflect",
            "window": "hann",
            "batch_size": 8,
            "context": [],
            "nodes": None,
        },
    }
    assert config["postprocessor"]["name"] == "segmentation"
    assert config["ingestor"] is None
    assert config["pipeline"] == "raster"


def test_a_released_pipeline_reloads_and_predicts_the_same(tmp_path):
    pipeline = build_library_pipeline()
    scene = build_scene(times=None)
    release = ChainModel.from_pipeline(pipeline)

    release.save_pretrained(tmp_path / "release")
    reloaded = load_pipeline(str(tmp_path / "release"))

    expected = pipeline.predict(as_rasters(scene))
    product = reloaded.predict(as_rasters(scene))
    assert product["class_id"].values.tolist() == expected["class_id"].values.tolist()
    # A reloaded pipeline holds the native chain, so it can be released again.
    assert isinstance(reloaded.model, ModelChain)
    again = ChainModel.from_pipeline(reloaded)
    assert again.config.predictor == release.config.predictor


def test_automodel_loads_the_model_of_a_released_pipeline_alone(tmp_path):
    pipeline = build_library_pipeline()
    ChainModel.from_pipeline(pipeline).save_pretrained(tmp_path / "release")

    model = AutoModel.from_pretrained(tmp_path / "release")

    assert isinstance(model, ChainModel)
    image = torch.zeros(1, 2, 16, 16)
    with torch.inference_mode():
        expected = pipeline.model.eval()(image=image)
        logits = model.eval()(image=image)
    torch.testing.assert_close(logits, expected)


class Doubled(RasterPreprocessor):
    """A project's own preprocessor, importable where the project is installed."""

    def __init__(self, bands):
        self.bands = bands

    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        image = rasters["image"].to_dataset()[self.bands].astype("float32")
        sample = xr.DataTree.from_dict({"image": image * 2})
        return sample


def build_project_pipeline() -> RasterPipeline:
    return RasterPipeline(
        _segmentation_chain(),
        preprocessor=Doubled(BANDS),
        predictor=DensePredictor(size=16, stride=12),
        postprocessor=RegressionPostprocessor(names=["a", "b", "c"]),
    )


def test_a_project_sub_processor_is_released_by_class_path_even_when_registered(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(PREPROCESSORS, "_factories", dict(PREPROCESSORS))
    PREPROCESSORS.register("doubled")(Doubled)

    release = ChainModel.from_pipeline(build_project_pipeline())

    assert release.config.preprocessor == {
        "class_path": f"{__name__}.Doubled",
        "init_args": {"bands": BANDS},
    }


def test_a_project_sub_processor_reloads_from_its_class_path(tmp_path):
    pipeline = build_project_pipeline()
    rasters = as_rasters(build_scene(times=None))

    ChainModel.from_pipeline(pipeline).save_pretrained(tmp_path / "release")
    reloaded = load_pipeline(str(tmp_path / "release"))

    assert isinstance(reloaded.preprocessor, Doubled)
    xr.testing.assert_allclose(reloaded.predict(rasters), pipeline.predict(rasters))


def test_a_sub_processor_that_cannot_be_imported_is_refused_at_release():
    class Local(Doubled):
        pass

    pipeline = RasterPipeline(
        _segmentation_chain(),
        preprocessor=Local(BANDS),
        predictor=DensePredictor(size=16),
        postprocessor=RegressionPostprocessor(names=["a", "b", "c"]),
    )

    with pytest.raises(ValueError, match="define it at module level"):
        ChainModel.from_pipeline(pipeline)


def test_a_release_naming_code_that_is_not_installed_says_which(tmp_path):
    release = ChainModel.from_pipeline(build_project_pipeline())
    release.config.preprocessor["class_path"] = "missing_project.Doubled"
    release.save_pretrained(tmp_path / "release")

    with pytest.raises(ModuleNotFoundError, match="missing_project"):
        load_pipeline(str(tmp_path / "release"))


class Stacked(ChangePreprocessor):
    def __call__(self, before: xr.DataTree, after: xr.DataTree) -> xr.DataTree:
        sample = xr.DataTree.from_dict({"image": before["image"].to_dataset()})
        return sample


def test_only_a_raster_pipeline_can_be_released():
    pipeline = ChangePipeline(
        _segmentation_chain(),
        Stacked(),
        DensePredictor(size=16),
        RegressionPostprocessor(names=["a", "b", "c"]),
    )

    with pytest.raises(TypeError, match="only a RasterPipeline can be released"):
        ChainModel.from_pipeline(pipeline)  # pyright: ignore[reportArgumentType]


def test_a_release_of_another_pipeline_variant_refuses_to_load(tmp_path):
    release = ChainModel.from_pipeline(build_library_pipeline())
    release.config.pipeline = "change"
    release.save_pretrained(tmp_path / "release")

    with pytest.raises(ValueError, match="carries a 'change' pipeline"):
        load_pipeline(str(tmp_path / "release"))


def test_a_release_without_a_pipeline_refuses_to_load_one(tmp_path):
    chain = build_model(
        {
            "head": {
                "name": "segmentation",
                "init_args": {"feature_channels": 2, "num_classes": 3},
            }
        }
    )
    ChainModel.from_chain(chain).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="carries no pipeline"):
        load_pipeline(str(tmp_path))
