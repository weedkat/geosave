"""Generated projects own their Lightning training classes."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from geosave.cli.core.templates import get_tasks
from geosave.cli.core.workspace import add_task, create_workspace
from geosave.ml.cli import GeosaveCLI
from modules.segmentation import supervised
from geosave.model.spec import ModelSpec, Ref


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path)
    add_task(tmp_path, "segmentation", "supervised")
    assert (tmp_path / "configs/segmentation-supervised.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_segmentation_configs_agree_on_model_inputs(workspace):
    spec = ModelSpec.load(workspace / "configs/segmentation-supervised/model_spec.yaml")
    config = yaml.safe_load(
        (workspace / "configs/segmentation-supervised.yaml").read_text()
    )
    requirement = spec.rasters["sentinel_2_l2a"]
    assert requirement.stac is not None
    assert requirement.stac.collection == "sentinel-2-l2a"
    assert requirement.variables == ("B02", "B03", "B04", "B08")
    assert tuple(map(str, requirement.stac.endpoints)) == (
        "planetary_computer",
        "cdse",
    )
    assert requirement.require_crs
    supervised.Module(**config["model"]["init_args"])
    encoder = config["model"]["init_args"]["model_chain"]["encoder"]["init_args"]
    assert encoder["in_channels"] == len(requirement.variables) == 4
    head = config["model"]["init_args"]["model_chain"]["head"]
    assert head == {
        "name": "segmentation",
        "init_args": {"num_classes": 2},
    }
    assert config["data"]["class_path"] == (
        "modules.segmentation.supervised.data.DataModule"
    )
    from modules.segmentation.supervised.datasets import MyDataset

    args = dict(config["data"]["init_args"])
    args.update(train=MyDataset("train"), val=MyDataset("val"))
    data = supervised.DataModule(**args)
    assert data.spec == spec
    assert spec.chips is not None
    assert spec.chips.shape == (encoder["input_size"],) * 2
    assert spec.pixel_inputs == ("image",)
    assert len(spec.transforms["image"][0].init_args["mean"]) == 4
    assert [step["name"] for step in config["data"]["init_args"]["augmentations"]] == [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
    ]
    assert not (workspace / "modules" / "data.py").exists()
    assert not (workspace / "configs" / "augmentation.yaml").exists()


def test_model_config_uses_current_task_constructor(workspace):
    config = yaml.safe_load(
        (workspace / "configs/segmentation-supervised.yaml").read_text()
    )
    model = supervised.Module(**config["model"]["init_args"])
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.optimizer_spec["name"] == "adamw"
    assert model.scheduler_spec["name"] == "cosine_annealing"


def test_generated_entrypoint_keeps_task_optimizer_ownership(workspace, monkeypatch):
    options = {}

    def observe_cli(self, **kwargs):
        options.update(kwargs)

    # Stop at the CLI boundary: entry-point execution must not start training.
    monkeypatch.setattr(GeosaveCLI, "__init__", observe_cli)
    runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert options["auto_configure_optimizers"] is False
    assert options["subclass_mode_model"] is True
    assert options["subclass_mode_data"] is True


def test_generated_entrypoint_parses_local_lightning_pair(
    workspace, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "fit",
            "--config",
            str(workspace / "configs/segmentation-supervised.yaml"),
            "--print_config",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert stopped.value.code == 0
    config = yaml.safe_load(capsys.readouterr().out)
    chain = config["model"]["init_args"]["model_chain"]
    assert chain["encoder"]["init_args"] == {"in_channels": 4, "input_size": 224}
    assert (
        config["data"]["init_args"]["spec"]
        == "configs/segmentation-supervised/model_spec.yaml"
    )
    assert config["data"]["class_path"] == (
        "modules.segmentation.supervised.data.DataModule"
    )


@pytest.mark.slow
def test_lightning_boilerplate_runs_one_train_and_validation_batch(tmp_path):
    assert "lightning" in get_tasks()
    create_workspace(tmp_path)
    add_task(tmp_path, "lightning", "boilerplate")
    assert (tmp_path / "configs/lightning-boilerplate.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    script = """
from modules.lightning.boilerplate.data import DataModule
from modules.lightning.boilerplate.module import Module
from geosave.ml.cli import GeosaveCLI

cli = GeosaveCLI(
    run=False,
    args=["--config", "configs/lightning-boilerplate.yaml"],
    save_config_callback=None,
    auto_configure_optimizers=False,
    subclass_mode_model=True,
    subclass_mode_data=True,
)
assert isinstance(cli.model, Module)
assert isinstance(cli.datamodule, DataModule)
cli.datamodule.setup("fit")
train_batch = next(iter(cli.datamodule.train_dataloader()))
validation_batch = next(iter(cli.datamodule.val_dataloader()))
assert cli.model.training_step(train_batch, 0).ndim == 0
assert cli.model.validation_step(validation_batch, 0).ndim == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def regression_workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path)
    add_task(tmp_path, "regression", "supervised")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_regression_swaps_only_the_head_loss_metrics_and_target(
    regression_workspace,
):
    from modules.regression import supervised as regression

    config = yaml.safe_load(
        (regression_workspace / "configs/regression-supervised.yaml").read_text()
    )
    chain = config["model"]["init_args"]["model_chain"]
    assert chain["head"] == {
        "name": "regression",
        "init_args": {"variables": ["target"]},
    }
    assert chain["encoder"]["name"] == "dinov3"
    module = regression.Module(**config["model"]["init_args"])
    assert isinstance(module.criterion, torch.nn.MSELoss)
    data = config["data"]
    assert data["class_path"] == "modules.regression.supervised.data.DataModule"
    assert data["init_args"]["spec"] == "configs/regression-supervised/model_spec.yaml"
    assert set(data["init_args"]["train"]) == {"class_path", "init_args"}


def test_shipped_model_spec_round_trips_as_inert_declarations(tmp_path):
    path = (
        Path(__file__).parents[3]
        / "src/templates/tasks/segmentation/supervised/configs/segmentation-supervised/model_spec.yaml"
    )

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path / "model_spec.yaml"))

    assert restored == spec
    assert spec.rasters["sentinel_2_l2a"].variables == (
        "B02",
        "B03",
        "B04",
        "B08",
    )
    assert spec.preprocessing["image"].call == Ref("sentinel_2_l2a.gs.mask_and_scale")
    assert spec.preprocessing["image"].kwargs == {}


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    create_workspace(tmp_path)
    add_task(tmp_path, "segmentation", "supervised")

    result = subprocess.run(
        [sys.executable, "scripts/segmentation/supervised/prepare_example.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "B08" in result.stdout
    assert "0.6" in result.stdout


@pytest.mark.slow
def test_every_recipe_instantiates_from_its_generated_config(tmp_path):
    from geosave.cli.core.templates import get_methods

    create_workspace(tmp_path)
    for task in get_tasks():
        for method in get_methods(task):
            add_task(tmp_path, task, method)
    script = """
from pathlib import Path
import sys
import yaml
from geosave.ml.cli import GeosaveCLI

for path in sorted(Path('configs').glob('*.yaml')):
    config = yaml.safe_load(path.read_text())
    config['trainer'] = {
        'accelerator': 'cpu', 'devices': 1, 'logger': False,
        'enable_checkpointing': False, 'enable_progress_bar': False,
    }
    config['data']['init_args']['num_workers'] = 0
    cli = GeosaveCLI(
        run=False, args=config, save_config_callback=None,
        auto_configure_optimizers=False,
        subclass_mode_model=True, subclass_mode_data=True,
    )
    assert type(cli.model).__module__.startswith('modules.')
    assert type(cli.datamodule).__module__.startswith('modules.')
    print(path.name)
assert not any(name.startswith('geosave.ml.tasks') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 7


@pytest.mark.slow
def test_generated_module_checkpoint_exports_matching_inference(tmp_path):
    create_workspace(tmp_path)
    add_task(tmp_path, "segmentation", "supervised")
    script = """
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from transformers import AutoModel
from modules.segmentation.supervised.module import Module
from geosave.model.release import GeoSaveModel

module = Module(model_chain={
    'head': {'name': 'segmentation', 'init_args': {
        'feature_channels': 2, 'num_classes': 2,
    }},
})
inputs = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32
samples = [{'inputs': {'feature_map': inputs}, 'target': torch.zeros(4, 4, dtype=torch.long)}] * 2
trainer = Trainer(
    accelerator='cpu', devices=1, max_steps=1, logger=False,
    enable_checkpointing=False, enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=2))
trainer.save_checkpoint('trained.ckpt')
restored = Module.load_from_checkpoint('trained.ckpt').eval()
release = GeoSaveModel.from_chain(restored.model).eval()
release.save_pretrained('release')
loaded = AutoModel.from_pretrained('release', local_files_only=True).eval()
module.eval()
with torch.inference_mode():
    expected = module(feature_map=inputs.unsqueeze(0))
    torch.testing.assert_close(restored(feature_map=inputs.unsqueeze(0)), expected)
    torch.testing.assert_close(loaded(feature_map=inputs.unsqueeze(0)), expected)
assert Path('release/model.safetensors').is_file()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
