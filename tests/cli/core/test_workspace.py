"""Generated projects own their Lightning training classes."""

import subprocess
import sys

import pytest

from geosave.cli.core.templates import get_tasks
from geosave.cli.core.workspace import add_task, create_workspace


@pytest.mark.slow
def test_lightning_boilerplate_runs_one_train_and_validation_batch(tmp_path):
    create_workspace(tmp_path)
    add_task(tmp_path, "lightning", "boilerplate")
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
    count = 0
    for task in get_tasks():
        for method in get_methods(task):
            add_task(tmp_path, task, method)
            count += 1
    script = """
from pathlib import Path
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
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == count


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
