"""Exercise the same project files and imports that a workspace user receives."""

import subprocess
import sys

import pytest

from geosave.cli.core import templates
from geosave.cli.core.workspace import create_workspace


def test_overlapping_starter_sources_are_reported_before_writing(tmp_path, monkeypatch):
    common = tmp_path / "common"
    tasks = tmp_path / "tasks"
    task_common = tasks / "test/common"
    recipe = tasks / "test/methods/example"
    for directory in (common, task_common, recipe):
        directory.mkdir(parents=True)
    (common / "main.py").write_text("common")
    (recipe / "description.txt").write_text("Example")
    (task_common / "main.py").write_text("overlap")
    monkeypatch.setattr(templates, "COMMON_DIR", common)
    monkeypatch.setattr(templates, "TASKS_DIR", tasks)
    root = tmp_path / "demo"
    with pytest.raises(ValueError, match="main.py"):
        create_workspace(root, "test", "example")
    assert not root.exists()


def test_workspace_copies_binary_and_text_files_verbatim(tmp_path, monkeypatch):
    source = tmp_path / "common"
    source.mkdir()
    data = b"\xff\x00modules.segmentation.example"
    (source / "asset").write_bytes(data)
    monkeypatch.setattr(templates, "COMMON_DIR", source)
    root = tmp_path / "demo"
    create_workspace(root)
    assert (root / "asset").read_bytes() == data


@pytest.mark.slow
def test_lightning_boilerplate_runs_one_train_and_validation_batch(tmp_path):
    create_workspace(tmp_path, "lightning", "boilerplate")
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "fit",
            "--config",
            "configs/boilerplate.yaml",
            "--trainer.fast_dev_run",
            "1",
            "--trainer.accelerator",
            "cpu",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    create_workspace(tmp_path, "segmentation", "supervised")
    result = subprocess.run(
        [sys.executable, "scripts/prepare_example.py"],
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
    script = """
from pathlib import Path
import yaml
from geosave.ml.torch.training.cli import GeosaveCLI

(path,) = Path("configs").glob("*.yaml")
config = yaml.safe_load(path.read_text())
config["trainer"] = {
    "accelerator": "cpu", "devices": 1, "logger": False,
    "enable_checkpointing": False, "enable_progress_bar": False,
}
config["data"]["init_args"]["num_workers"] = 0
cli = GeosaveCLI(
    run=False, args=config, save_config_callback=None,
    auto_configure_optimizers=False, subclass_mode_model=True, subclass_mode_data=True,
)
assert type(cli.model).__module__.startswith("training.")
assert type(cli.datamodule).__module__.startswith("training.")
"""
    for task in templates.get_tasks():
        for method in templates.get_methods(task):
            root = tmp_path / task / method
            create_workspace(root, task, method)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert result.returncode == 0, f"{task}/{method}: {result.stderr}"


@pytest.mark.slow
def test_generated_module_checkpoint_exports_matching_inference(tmp_path):
    create_workspace(tmp_path, "segmentation", "supervised")
    script = """
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from training import Module
from geosave.ml.torch import ModelChain

module = Module(
    model_chain={'head': {'name': 'segmentation', 'init_args': {
        'feature_channels': 2, 'num_classes': 2,
    }}},
    ignore_index=255,
    criterion=torch.nn.CrossEntropyLoss(ignore_index=255),
    optimizer=partial(torch.optim.AdamW, lr=1e-3),
    param_groups=None,
    lr_scheduler=None,
    lr_scheduler_config=None,
    metrics=['iou_macro'],
)
inputs = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32
grid = {'transform': torch.tensor([1.0, 0, 0, 0, -1, 0], dtype=torch.float64), 'crs': torch.tensor(32749)}
samples = [{'feature_map': inputs, 'target': torch.zeros(4, 4, dtype=torch.long), **grid}] * 2
trainer = Trainer(
    accelerator='cpu', devices=1, max_steps=1, logger=False,
    enable_checkpointing=False, enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=2))
trainer.save_checkpoint('trained.ckpt')
restored = Module.load_from_checkpoint('trained.ckpt').eval()
restored.model.save_pretrained('release')
loaded = ModelChain.from_pretrained('release').eval()
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
