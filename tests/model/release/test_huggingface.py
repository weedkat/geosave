"""Transformers is the sole persistence adapter for configured model chains."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from huggingface_hub.errors import EntryNotFoundError
import pytest
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModel

from geosave.__about__ import __version__
from geosave.model.chain import ModelChain, chain_step
from geosave.model.stages.head.segmentation import SegmentationHead
from geosave.model.registry import build_model, register_model
from geosave.model.release import GeoSaveConfig, GeoSaveModel
from geosave.model.spec import ModelSpec


@register_model("head", "periodic")
class PeriodicHead(torch.nn.Module):
    """Holds a non-persistent buffer filled by a method, as timm's RoPE does."""

    num_classes = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("periods", torch.empty(3), persistent=False)
        self.init_non_persistent_buffers()

    def init_non_persistent_buffers(self) -> None:
        self.periods.copy_(torch.tensor([1.0, 2.0, 4.0]))

    @chain_step(head=True)
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map * self.scale * self.periods.sum()


@pytest.fixture
def stages() -> dict[str, dict[str, object]]:
    return {
        "head": {
            "name": "segmentation",
            "init_args": {"feature_channels": 2, "classes": ["a", "b", "c"]},
        }
    }


@pytest.fixture
def chain(stages: dict[str, dict[str, object]]) -> ModelChain:
    chain = build_model(stages)
    with torch.no_grad():
        chain.get_submodule("head").layers[-1].weight.fill_(2.0)
    return chain


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(schema_version=2, rasters={})


def test_automodel_roundtrip_preserves_recipe_weights_outputs_and_mode(
    tmp_path: Path, chain: ModelChain
) -> None:
    published = GeoSaveModel.from_chain(chain)
    published.eval()
    inputs = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)

    published.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(restored, GeoSaveModel)
    assert not restored.training
    assert list(restored.chain.stage_specs) == ["head"]
    assert config["format_version"] == 1
    assert config["geosave_version"] == __version__
    assert "auto_map" not in config
    torch.testing.assert_close(
        restored(feature_map=inputs), published(feature_map=inputs)
    )
    for name, value in chain.state_dict().items():
        torch.testing.assert_close(restored.chain.state_dict()[name], value)


def test_wrapping_keeps_trained_weights(chain: ModelChain) -> None:
    trained = {name: value.clone() for name, value in chain.state_dict().items()}

    published = GeoSaveModel.from_chain(chain)

    assert published.chain is chain
    for name, value in trained.items():
        torch.testing.assert_close(chain.state_dict()[name], value)


def test_save_writes_the_spec_beside_the_weights(
    tmp_path: Path, chain: ModelChain, spec: ModelSpec
) -> None:
    GeoSaveModel.from_chain(chain, spec=spec).save_pretrained(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "config.json",
        "model.safetensors",
        ModelSpec.filename,
    }
    assert ModelSpec.load(tmp_path / ModelSpec.filename) == spec


def test_save_without_a_spec_writes_only_the_model(
    tmp_path: Path, chain: ModelChain
) -> None:
    GeoSaveModel.from_chain(chain).save_pretrained(tmp_path)

    assert not (tmp_path / ModelSpec.filename).exists()


def test_push_uploads_model_spec_and_card_in_one_commit(
    tmp_path: Path, chain: ModelChain, spec: ModelSpec
) -> None:
    uploaded: dict[str, str] = {}

    def upload(folder: str, *args: object, **kwargs: object) -> SimpleNamespace:
        uploaded.update(
            {
                path.name: path.read_text(errors="ignore")
                for path in Path(folder).iterdir()
            }
        )
        return SimpleNamespace(oid="commit-oid")

    api = MagicMock()
    api.create_repo.return_value.repo_id = "org/model"
    release = GeoSaveModel.from_chain(chain, spec=spec)
    with (
        patch("transformers.utils.hub.hf_api", return_value=api),
        patch(
            "transformers.utils.hub.ModelCard.load",
            side_effect=EntryNotFoundError("no card"),
        ),
        patch.object(GeoSaveModel, "_upload_modified_files", side_effect=upload),
    ):
        commit = release.push_to_hub("org/model")

    assert commit.oid == "commit-oid"
    assert set(uploaded) == {
        "README.md",
        "config.json",
        "model.safetensors",
        ModelSpec.filename,
    }
    assert "geosave" in uploaded["README.md"]


def test_reload_fills_non_persistent_buffers(tmp_path: Path) -> None:
    chain = build_model({"head": {"name": "periodic"}})
    GeoSaveModel.from_chain(chain).save_pretrained(tmp_path)

    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)

    torch.testing.assert_close(
        restored.chain.get_submodule("head").periods, torch.tensor([1.0, 2.0, 4.0])
    )


def test_export_rejects_a_directly_composed_chain() -> None:
    chain = ModelChain(
        head=SegmentationHead(feature_channels=2, classes=["a", "b", "c"])
    )

    with pytest.raises(ValueError, match="stage specifications"):
        GeoSaveModel.from_chain(chain)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "Missing keys"),
        ("unexpected", "Unexpected keys"),
        ("mismatched", "size mismatch|ignore_mismatched_sizes"),
    ],
)
def test_reload_rejects_incompatible_weights(
    tmp_path: Path,
    chain: ModelChain,
    change: str,
    message: str,
) -> None:
    GeoSaveModel.from_chain(chain).save_pretrained(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    state = load_file(weights_path)
    if change == "missing":
        state.pop(next(iter(state)))
    elif change == "unexpected":
        state["unexpected"] = torch.tensor(1.0)
    else:
        key = next(iter(state))
        state[key] = torch.zeros(1)
    save_file(state, weights_path)

    with pytest.raises(RuntimeError, match=message):
        AutoModel.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.slow
def test_automodel_loads_in_fresh_process(tmp_path: Path, chain: ModelChain) -> None:
    published = GeoSaveModel.from_chain(chain)
    published.eval()
    inputs = torch.ones(1, 2, 4, 4)
    torch.save(published(feature_map=inputs).detach(), tmp_path / "expected.pt")
    published.save_pretrained(tmp_path / "export")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import geosave.model.release
from transformers import AutoModel
model = AutoModel.from_pretrained(sys.argv[1], local_files_only=True)
actual = model(feature_map=torch.ones(1, 2, 4, 4))
expected = torch.load(sys.argv[2], weights_only=True)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
""",
            str(tmp_path / "export"),
            str(tmp_path / "expected.pt"),
        ],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_publication_rejects_class_path_stages() -> None:
    chain = build_model(
        {
            "head": {
                "class_path": "tests.model.chain.conftest.Head",
                "init_args": {"channels": 2},
            }
        }
    )

    with pytest.raises(ValueError, match="registered name"):
        GeoSaveModel.from_chain(chain)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 99, "format version"),
        ("geosave_version", "0.0.0", "GeoSave version"),
    ],
)
def test_loading_rejects_incompatible_artifact_metadata(
    tmp_path: Path,
    chain: ModelChain,
    field: str,
    value: object,
    message: str,
) -> None:
    GeoSaveModel.from_chain(chain).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))

    with pytest.raises((ValueError, RuntimeError), match=message):
        GeoSaveModel.from_pretrained(tmp_path, local_files_only=True)


def test_config_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        GeoSaveConfig(
            stages=[
                {"stage": "encoder", "spec": {"name": "segmentation"}},
                {"stage": "encoder", "spec": {"name": "segmentation"}},
            ]
        )
