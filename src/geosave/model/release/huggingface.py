"""Persist native model chains through the Transformers file contract."""

from __future__ import annotations

import os
from typing import Any, TypedDict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from geosave.__about__ import __version__
from geosave.model.chain import ModelChain
from geosave.model.registry import build_model
from geosave.model.spec import ModelSpec

ARTIFACT_FORMAT_VERSION = 1


class StageConfig(TypedDict):
    """One named stage in an ordered Transformers configuration.

    Attributes:
        stage: Chain module name and registry stage, such as `encoder` or
            `head`. Stage names must be unique in a release configuration.
        spec: Factory selector and resolved `init_args` used to reconstruct
            the module. Publication uses registered `name` specifications.
    """

    stage: str
    spec: dict[str, Any]


class GeoSaveConfig(PretrainedConfig):
    """Describe a native chain in a generated model artifact.

    Args:
        stages: Ordered stage names and construction specifications.
        format_version: GeoSave release format version.
        geosave_version: Exact GeoSave version required to load the artifact.
        **kwargs: Standard Transformers configuration fields.

    Raises:
        ValueError: The format version is unsupported or stage names repeat.
        RuntimeError: The artifact requires a different GeoSave version.
    """

    model_type = "geosave_chain"

    def __init__(
        self,
        stages: list[StageConfig] | None = None,
        format_version: int = ARTIFACT_FORMAT_VERSION,
        geosave_version: str = __version__,
        **kwargs: Any,
    ) -> None:
        if format_version != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported GeoSave artifact format version {format_version}; "
                f"expected {ARTIFACT_FORMAT_VERSION}"
            )
        if geosave_version != __version__:
            raise RuntimeError(
                f"Artifact requires GeoSave version {geosave_version}; "
                f"installed version is {__version__}"
            )
        super().__init__(**kwargs)
        self.stages = stages if stages is not None else []
        self.format_version = format_version
        self.geosave_version = geosave_version
        names = [entry["stage"] for entry in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Each stage must have a unique name")


class GeoSaveModel(PreTrainedModel):
    """Load, save, and push a native model chain as a Transformers model.

    Args:
        config: Ordered construction specifications.
        chain: Existing chain to wrap without rebuilding or copying weights.
            Its specifications must match config. Omit when loading an artifact.
        spec: Processing contract `save_pretrained` writes beside the weights.

    Raises:
        ValueError: The supplied chain does not match the configuration.

    Examples:
        >>> release = GeoSaveModel.from_chain(task.model, spec=spec)
        >>> commit = release.push_to_hub("org/model")
        >>> model = AutoModel.from_pretrained("org/model", revision=commit.oid)
        >>> model.chain.encoder
        DINOv3(...)
    """

    config_class = GeoSaveConfig
    base_model_prefix = "chain"

    def __init__(
        self,
        config: GeoSaveConfig,
        *,
        chain: ModelChain | None = None,
        spec: ModelSpec | None = None,
    ) -> None:
        super().__init__(config)
        specs = {entry["stage"]: entry["spec"] for entry in config.stages}
        if chain is not None and list(chain.stage_specs.items()) != list(specs.items()):
            raise ValueError("Chain construction specifications do not match config")
        self.chain = chain if chain is not None else build_model(specs)
        self.spec = spec
        self.add_model_tags("geosave")
        self.post_init()

    @classmethod
    def from_chain(
        cls, chain: ModelChain, *, spec: ModelSpec | None = None
    ) -> GeoSaveModel:
        """Wrap a trained chain, sharing its modules and weights.

        Args:
            chain: Model built from registered stage specifications.
            spec: Processing contract to release with the weights.

        Returns:
            Transformers model sharing the chain and its training mode.

        Raises:
            ValueError: The chain has no construction specifications, or a
                stage was built from a class path.
        """
        specs = chain.stage_specs
        for stage, stage_spec in specs.items():
            if not stage_spec.get("name") or "class_path" in stage_spec:
                raise ValueError(
                    f"Stage {stage!r} must use a registered name for publication"
                )
        config = GeoSaveConfig(
            stages=[
                StageConfig(stage=name, spec=value) for name, value in specs.items()
            ]
        )
        model = cls(config, chain=chain, spec=spec)
        model.training = chain.training
        return model

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Write the Transformers artifact and, when present, the spec beside it.

        Args:
            save_directory: Directory to write.
            is_main_process: Whether this process writes; only it writes the spec.
            *args: Further positional options of `PreTrainedModel.save_pretrained`.
            **kwargs: Options forwarded to `PreTrainedModel.save_pretrained`.
        """
        super().save_pretrained(save_directory, is_main_process, *args, **kwargs)
        if self.spec is not None and is_main_process:
            self.spec.save(os.path.join(save_directory, ModelSpec.filename))

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Fill derived buffers; parameters come from stage constructors or weights.

        Transformers loads into modules built on the meta device, which leaves
        non-persistent buffers (such as timm's rotary periods) unfilled.
        """
        fill = getattr(module, "init_non_persistent_buffers", None)
        if callable(fill):
            fill()

    @staticmethod
    def _finalize_model_loading(
        model: PreTrainedModel, load_config: Any, loading_info: Any
    ) -> Any:
        """Reject checkpoints whose keys differ from the configured chain."""
        loading_info = PreTrainedModel._finalize_model_loading(
            model, load_config, loading_info
        )
        problems = [
            f"{label} keys: {sorted(keys)}"
            for label, keys in (
                ("Missing", loading_info.missing_keys),
                ("Unexpected", loading_info.unexpected_keys),
            )
            if keys
        ]
        if problems:
            raise RuntimeError(
                "GeoSaveModel requires an exact checkpoint match. "
                + "; ".join(problems)
            )
        return loading_info

    def forward(self, **inputs: object) -> object:
        """Evaluate the chain using named tensor and geospatial context inputs.

        Args:
            **inputs: Values required by the chain's selected methods.

        Returns:
            The chain's predictions or feature context.
        """
        return self.chain(**inputs)


AutoConfig.register(GeoSaveConfig.model_type, GeoSaveConfig)
AutoModel.register(GeoSaveConfig, GeoSaveModel)
