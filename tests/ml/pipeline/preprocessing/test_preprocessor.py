import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from jsonargparse import ArgumentParser

from geosave.ml.pipeline import IngestOutput
from geosave.ml.pipeline.preprocessing import Composite, MaskAndScale, Preprocessor
from tests.geodata.conftest import _refuse
from tests.geodata.sampling.conftest import GRID, build_scene


def sources(
    dates: tuple[str, ...] = ("2024-07-08", "2024-07-22", "2024-08-05"),
) -> xr.DataTree:
    """Ingested tree: packed reflectance observed on `dates`, on a 10 x 10 grid."""
    scene = (build_scene(GRID, pd.to_datetime(list(dates))) * 10).astype("uint16")
    for variable in scene.data_vars.values():
        variable.attrs.update(
            scale_factor=np.float32(0.1),
            add_offset=np.float32(0),
            _FillValue=np.uint16(0),
        )
    return xr.DataTree.from_dict({"s2/reflectance": scene})


def window_times(window: xr.DataTree) -> list[str]:
    return window["s2/reflectance"].time.dt.strftime("%m-%d").values.tolist()


def test_steps_run_in_order_then_each_window_is_one_lazy_input():
    preprocessor = Preprocessor(
        [MaskAndScale(), Composite("MS", "median")], time_freq="MS"
    )

    with dask.config.set(scheduler=_refuse):
        windows = preprocessor(IngestOutput(sources())).windows

    assert [window_times(window) for window in windows] == [["07-01"], ["08-01"]]
    reflectance = windows[0]["s2/reflectance"].to_dataset()
    assert reflectance.red.dtype == "float32"
    expected = sources().gs.mask_and_scale().gs.composite("MS", "median")
    xr.testing.assert_allclose(
        reflectance.compute(),
        expected["s2/reflectance"].to_dataset().isel(time=[0]).compute(),
    )


def test_without_time_freq_the_prepared_tree_is_the_one_input():
    (prepared,) = Preprocessor([MaskAndScale()])(IngestOutput(sources())).windows

    xr.testing.assert_identical(prepared, sources().gs.mask_and_scale())


def test_windows_follow_size_stride_and_keep():
    preprocessor = Preprocessor(
        [], time_freq="MS", time_window=2, time_stride=2, time_keep="first"
    )
    dates = ("2024-07-08", "2024-07-22", "2024-08-05", "2024-09-02", "2024-10-07")

    windows = preprocessor(IngestOutput(sources(dates))).windows

    assert [window_times(window) for window in windows] == [
        ["07-08", "08-05"],
        ["09-02", "10-07"],
    ]


def test_a_window_with_an_empty_slot_follows_missing():
    skipping = Preprocessor([], time_freq="MS", time_window=2)
    raising = Preprocessor([], time_freq="MS", time_window=2, time_missing="raise")

    # Every two-month window between July and September has an empty August.
    gap = IngestOutput(sources(("2024-07-08", "2024-09-05")))

    with pytest.raises(ValueError, match="no complete temporal window"):
        skipping(gap)
    with pytest.raises(ValueError, match="empty"):
        raising(gap)


def test_a_config_builds_the_preprocessor_and_writes_out_every_argument():
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--preprocessor", type=Preprocessor)
    config = parser.parse_object(
        {
            "preprocessor": {
                "steps": [
                    {"class_path": "geosave.ml.pipeline.preprocessing.MaskAndScale"},
                    {
                        "class_path": "geosave.ml.pipeline.preprocessing.Composite",
                        "init_args": {"freq": "MS", "method": "median"},
                    },
                ],
                "time_freq": "MS",
            }
        }
    )

    preprocessor = parser.instantiate_classes(config).preprocessor

    assert [type(step) for step in preprocessor.steps] == [MaskAndScale, Composite]
    assert len(preprocessor(IngestOutput(sources())).windows) == 2
    dumped = yaml.safe_load(parser.dump(config))["preprocessor"]["init_args"]
    composite = dumped["steps"][1]["init_args"]
    assert composite["origin"] == "start_day"
    assert dumped["time_missing"] == "skip"


def test_a_config_refuses_an_unknown_reducer_before_building():
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--step", type=Composite)

    with pytest.raises(Exception, match="medain"):
        parser.parse_object(
            {
                "step": {
                    "class_path": "geosave.ml.pipeline.preprocessing.Composite",
                    "init_args": {"freq": "MS", "method": "medain"},
                }
            }
        )
