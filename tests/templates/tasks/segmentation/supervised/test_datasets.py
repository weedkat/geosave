from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr
from lightning.pytorch import Trainer
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.model.spec import ModelSpec
from geosave.model.chain import chain_step
from geosave.model.processing import to_raster
from modules.segmentation import supervised
from modules.segmentation.supervised import datasets


class Segmenter(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv2d(4, num_classes, 1)

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


@pytest.fixture
def files(tmp_path):
    grid = GeoBox.from_bbox((0, 0, 80, 80), "EPSG:32633", resolution=10)
    image = xr.Dataset()
    # Store a different order from the recipe to detect positional selection.
    for band, value in [("B08", 4000), ("B04", 3000), ("B03", 2000), ("B02", 1000)]:
        image[band] = xr.DataArray(
            np.full((8, 8), value, dtype="uint16"), dims=("y", "x")
        )
        image[band].attrs.update(scale_factor=0.0001, _FillValue=65535)
    image = image.gs.write_grid(grid)
    target = xr.Dataset(
        {"label": (("y", "x"), np.zeros((8, 8), dtype="uint8"))}
    ).gs.write_grid(grid)
    target.label.values[:, 4:] = 1
    target.label.values[0, 0] = 99
    target.label.attrs["_FillValue"] = 99
    image.gs.to_cog(tmp_path, split_asset=(), format_paths="image.tif")
    target.gs.to_cog(tmp_path, split_asset=(), format_paths="label.tif")
    spec = ModelSpec.load(
        Path(datasets.__file__).parents[3]
        / "configs/segmentation-supervised/model_spec.yaml"
    )
    return image, target, spec


def open_dataset(tmp_path, spec):
    return datasets.RasterDataset(
        images=[str(tmp_path / "image.tif")],
        targets=[str(tmp_path / "label.tif")],
        spec=spec,
        num_classes=2,
    )


def test_raster_pairs_train_reload_and_predict_without_catalog(tmp_path, files):
    image, _, spec = files
    samples = open_dataset(tmp_path, spec)
    sample = samples[0]
    assert sample["inputs"]["image"].shape == (4, 8, 8)
    torch.testing.assert_close(
        sample["inputs"]["image"][:, 2, 2], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    assert sample["target"][0, 0] == 255
    assert sample["target"].dtype == torch.int64
    data = supervised.DataModule(
        spec, samples, samples, size=8, batch_size=1, num_workers=0
    )
    module = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.Segmenter",
                "init_args": {"num_classes": 2},
            }
        }
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 1
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = supervised.Module.load_from_checkpoint(checkpoint)
    restored.eval()
    saved = ModelSpec.load(spec.save(tmp_path / "recipe.yaml"))
    with gs.open_raster(tmp_path / "image.tif") as raster:
        inputs = {
            "image": saved.run(entry="encoded", output="normalized", raster=raster)
        }
    data.trainer = SimpleNamespace(training=False)
    batch = data.on_after_batch_transfer(next(iter(data.val_dataloader())), 0)
    torch.testing.assert_close(batch["inputs"]["image"][0], inputs["image"])
    with torch.no_grad():
        prediction = restored(**{"image": inputs["image"].unsqueeze(0)})
        torch.testing.assert_close(prediction, module(**batch["inputs"]))
    logits = to_raster(
        prediction[0], reference=image, dims=["band", "y", "x"], name="logits"
    )
    output = logits.argmax("band").rename("classes")
    output.gs.to_cog(tmp_path, format_paths="prediction.tif")
    with gs.open_raster(tmp_path / "prediction.tif") as reread:
        assert reread.gs.geobox == image.gs.geobox
        np.testing.assert_array_equal(reread.classes.values, output.values)


@pytest.mark.parametrize(
    "problem", ["grid", "fraction", "class", "time", "image_nodata"]
)
def test_raster_pairs_reject_unsafe_training_data(tmp_path, files, problem):
    image, target, spec = files
    message = ""
    if problem == "grid":
        target = target.gs.write_grid(
            GeoBox.from_bbox((10, 0, 90, 80), "EPSG:32633", resolution=10)
        )
        message = "grid"
    elif problem == "fraction":
        target = target.astype("float32")
        target.label.values[1, 1] = 0.5
        message = "integer"
    elif problem == "class":
        target.label.values[1, 1] = 2
        message = "class"
    elif problem == "image_nodata":
        image.B02.values[1, 1] = 65535
        image.gs.to_cog(
            tmp_path, split_asset=(), format_paths="image.tif", overwrite=True
        )
        message = "finite"
    else:
        # Validate the same selected-raster requirement without file splitting by time.
        with pytest.raises(ValueError, match="dim"):
            spec.run(
                entry="from_raster",
                output="prepared",
                raster=image.expand_dims(time=[0, 1]),
            )
        return
    target.gs.to_cog(tmp_path, split_asset=(), format_paths="label.tif", overwrite=True)
    with pytest.raises(ValueError, match=message):
        open_dataset(tmp_path, spec)[0]


def test_generated_normalization_preserves_input_gradients(files):
    _, _, spec = files
    image = torch.ones((4, 2, 2), requires_grad=True)
    prepared = image * 2
    outputs = spec.run(
        entry="from_tensor", output=["from_tensor", "normalized"], value=prepared
    )
    assert outputs["from_tensor"] is prepared
    outputs["normalized"].sum().backward()
    expected = 2 / torch.tensor([0.1809, 0.1757, 0.1888, 0.1742])
    torch.testing.assert_close(image.grad[:, 0, 0], expected)
