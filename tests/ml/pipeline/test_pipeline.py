from pathlib import Path

import dask
import numpy as np
import pytest
import yaml
import xarray as xr
from huggingface_hub import ModelHubMixin
from jsonargparse import ArgumentError

from geosave.geodata import GeoRegion, RasterChipper
from geosave.ml.pipeline import Ingestor, Pipeline, Preprocessor, Sentinel2L2AIngestor
from geosave.ml.pipeline.postprocessing import SegmentationPostprocessor
from geosave.ml.pipeline.preprocessing import MaskAndScale
from geosave.ml.pipeline.preprocessing.steps import Composite, Step
from geosave.ml.pipeline.prediction import ChipOutputs, PredictOutput, Predictor
from geosave.ml.pipeline.ingestion.sentinel2_l2a import SCL_CLASSES
from tests.geodata.conftest import build_raster
from tests.ml.pipeline.ingestion.test_sentinel2_l2a import (
    REGION,
    create_region_ingestor,
)


class Elevation(Ingestor):
    """A static collection: one timeless node on its own grid."""

    def __call__(self, region: GeoRegion) -> xr.DataTree:
        tree = build_raster()[["red"]].rename(red="height").gs.to_tree("dem")
        return tree.chunk()


class TopClass(ModelHubMixin):
    """A framework-free stand-in model: the class every chip scores highest.

    The Hub mixin writes `class_index` to `config.json` and reads it back.
    """

    def __init__(self, class_index: int = 1) -> None:
        self.class_index = class_index

    def _save_pretrained(self, save_directory: Path) -> None:
        pass

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id,
        revision,
        cache_dir,
        force_download,
        local_files_only,
        token,
        **model_kwargs,
    ) -> "TopClass":
        return cls(**model_kwargs)


class Constant(Predictor):
    """A framework-free stand-in predictor: every chip scores its model's class highest."""

    def __init__(self, model: TopClass) -> None:
        self.model = model

    def __call__(self, preprocessed) -> PredictOutput[np.ndarray]:
        chips = RasterChipper(preprocessed.windows, size=2)
        return PredictOutput(chips, self.score(chips))

    def score(self, chips: RasterChipper) -> ChipOutputs[np.ndarray]:
        for location, _ in chips:
            scores = np.zeros((2, 2, 2), dtype="float32")
            scores[self.model.class_index] = 1.0
            yield location, scores


def create_pipeline(*ingestors: Ingestor, **preprocessing) -> Pipeline:
    return Pipeline(
        list(ingestors),
        Preprocessor([], **preprocessing),
        Constant(TopClass()),
        SegmentationPostprocessor(),
    )


def create_monthly_pipeline(tmp_path, **preprocessing) -> Pipeline:
    """Two Sentinel-2 acquisitions, 2024-12-31 and 2025-01-01: two monthly windows."""
    return create_pipeline(
        create_region_ingestor(tmp_path), time_freq="MS", **preprocessing
    )


def test_without_a_store_every_collection_is_loaded_into_one_root(tmp_path):
    pipeline = create_pipeline(create_region_ingestor(tmp_path), Elevation())

    sources = pipeline.ingest(REGION).sources

    assert list(sources.children) == ["s2", "dem"]
    assert list(sources["s2"].children) == ["image", "scl"]
    assert isinstance(sources["s2/image"].to_dataset().B04.data, np.ndarray)


def test_two_collections_under_one_name_are_refused():
    pipeline = create_pipeline(Elevation(), Elevation())

    with pytest.raises(ValueError, match="'dem'"):
        pipeline.ingest(REGION)


def test_staged_sources_read_only_the_store(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pipeline = create_pipeline(create_region_ingestor(source), Elevation())
    expected = pipeline.ingest(REGION).sources

    stored = pipeline.ingest(REGION, store=tmp_path / "run").sources
    # The staged tree must not depend on the acquisitions' files any more.
    for path in source.iterdir():
        path.unlink()

    for node in ("s2/image", "s2/scl", "dem"):
        dataset = stored[node].to_dataset().compute()
        xr.testing.assert_identical(dataset, expected[node].to_dataset())
    image = stored["s2/image"].to_dataset()
    assert isinstance(image.B04.data, dask.array.Array)
    assert image.B04.chunks == ((1, 1), (2, 2), (2, 2))
    assert image.B04.gs.packing.add_offset == -0.1
    assert stored["s2/scl"].to_dataset().SCL.gs.flags.classes == SCL_CLASSES
    assert [path.name for path in (tmp_path / "run").iterdir()] == ["sources.zarr"]


def test_an_existing_store_is_not_overwritten(tmp_path):
    pipeline = create_pipeline(create_region_ingestor(tmp_path))
    pipeline.ingest(REGION, store=tmp_path / "run")

    with pytest.raises(FileExistsError):
        pipeline.ingest(REGION, store=tmp_path / "run")


def test_the_store_keeps_each_collection_time_chunks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pipeline = create_pipeline(create_region_ingestor(source, time_chunks=-1))

    stored = pipeline.ingest(REGION, store=tmp_path / "run").sources

    assert stored["s2/image"].to_dataset().B04.chunks == ((2,), (2, 2), (2, 2))


def test_the_product_stacks_every_window_along_time(tmp_path):
    pipeline = create_monthly_pipeline(tmp_path)

    product = pipeline(REGION).product

    assert product.class_id.dims == ("time", "y", "x")
    assert (product.class_id.values == 1).all()
    assert product.time.dt.strftime("%Y-%m-%d").values.tolist() == [
        "2024-12-31",
        "2025-01-01",
    ]
    start, end = product.gs.timespan
    assert (start.isoformat(), end.isoformat()) == (
        "2024-12-31T00:00:00",
        "2025-01-01T00:00:00",
    )


def test_a_staged_run_equals_the_run_in_memory_and_stays_lazy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pipeline = create_monthly_pipeline(source)
    expected = pipeline(REGION).product

    product = pipeline(REGION, store=tmp_path / "run").product

    assert isinstance(product.class_id.data, dask.array.Array)
    xr.testing.assert_identical(product.compute(), expected)
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "product",
        "sources.zarr",
    ]
    assert sorted(path.name for path in (tmp_path / "run/product").iterdir()) == [
        "0.zarr",
        "1.zarr",
    ]


def test_a_pipeline_without_time_windows_keeps_a_timeless_product():
    pipeline = create_pipeline(Elevation())

    product = pipeline(REGION).product

    assert product.class_id.dims == ("y", "x")
    assert product.gs.geobox == build_raster().gs.geobox


def test_a_region_without_a_complete_window_is_refused(tmp_path):
    pipeline = create_monthly_pipeline(tmp_path, time_window=3)

    with pytest.raises(ValueError, match="no complete temporal window"):
        pipeline(REGION)


CONFIG = {
    "ingestors": [
        {
            "class_path": "geosave.ml.pipeline.Sentinel2L2AIngestor",
            "init_args": {"name": "image", "bands": ["B04", "B08"]},
        }
    ],
    "preprocessor": {
        "steps": [{"class_path": "geosave.ml.pipeline.preprocessing.MaskAndScale"}],
        "time_freq": "MS",
    },
    "predictor": {"class_path": "tests.ml.pipeline.test_pipeline.Constant"},
    "postprocessor": {
        "class_path": "geosave.ml.pipeline.postprocessing.SegmentationPostprocessor",
        "init_args": {"window": "hann"},
    },
}


def test_from_config_builds_the_same_pipeline_from_a_mapping_or_a_file(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(CONFIG))

    for config in (CONFIG, path):
        pipeline = Pipeline.from_config(config, TopClass())
        assert [ingestor.bands for ingestor in pipeline.ingestors] == [["B04", "B08"]]
        assert pipeline.preprocessor.time_freq == "MS"
        assert isinstance(pipeline.predictor, Constant)
        assert pipeline.postprocessor.window == "hann"


def test_from_config_refuses_a_stage_of_the_wrong_kind():
    config = dict(CONFIG, predictor=CONFIG["postprocessor"])

    with pytest.raises(Exception, match="predictor"):
        Pipeline.from_config(config, TopClass())


class Normalize(Step):
    """A step whose arguments a config holds only as lists."""

    def __init__(self, mean: list[float]) -> None:
        self.mean = mean

    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        return tree


def create_released_pipeline() -> Pipeline:
    return Pipeline(
        ingestors=[Sentinel2L2AIngestor("image", ["B04", "B08"], resolution=20.0)],
        preprocessor=Preprocessor(
            [MaskAndScale(), Composite("MS", "median"), Normalize([0.1, 0.2])],
            time_freq="MS",
        ),
        predictor=Constant(TopClass(0)),
        postprocessor=SegmentationPostprocessor(window="hann"),
    )


def test_a_pipeline_built_from_instances_rebuilds_from_its_config():
    pipeline = create_released_pipeline()

    config = pipeline.to_config()
    rebuilt = Pipeline.from_config(config, pipeline.predictor.model)

    assert rebuilt.to_config() == config
    steps = config["preprocessor"]["init_args"]["steps"]
    assert [step["class_path"].rpartition(".")[2] for step in steps] == [
        "MaskAndScale",
        "Composite",
        "Normalize",
    ]
    assert config["ingestors"][0]["init_args"]["provider"] == "planetary_computer"
    assert "model" not in config["predictor"]["init_args"]


def test_the_config_reads_attributes_changed_after_construction():
    pipeline = create_released_pipeline()
    postprocessor = pipeline.postprocessor
    assert isinstance(postprocessor, SegmentationPostprocessor)
    postprocessor.window = "boxcar"

    config = pipeline.to_config()

    assert config["postprocessor"]["init_args"]["window"] == "boxcar"


def test_a_value_no_config_holds_is_refused():
    pipeline = create_released_pipeline()
    normalize = pipeline.preprocessor.steps[2]
    assert isinstance(normalize, Normalize)
    normalize.mean = np.array([0.1, 0.2])  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError, match="ndarray"):
        pipeline.to_config()


def test_a_value_that_does_not_fit_its_type_hint_is_refused():
    pipeline = create_released_pipeline()
    normalize = pipeline.preprocessor.steps[2]
    assert isinstance(normalize, Normalize)
    normalize.mean = "0.1"  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(ArgumentError, match="mean"):
        pipeline.to_config()


def test_a_stage_defined_in_main_is_refused(monkeypatch):
    pipeline = create_released_pipeline()
    monkeypatch.setattr(Normalize, "__module__", "__main__")

    with pytest.raises(ValueError, match="__main__"):
        pipeline.to_config()


def test_a_released_pipeline_loads_back_with_its_model(tmp_path):
    pipeline = create_released_pipeline()

    pipeline.save_pretrained(tmp_path)
    loaded = Pipeline.from_pretrained(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "README.md",
        "config.json",
        "pipeline.yaml",
    ]
    assert loaded.to_config() == pipeline.to_config()
    assert isinstance(loaded.predictor.model, TopClass)
    assert loaded.predictor.model.class_index == 0
