"""Persist native model chains and their processors through the Transformers file contract.

A release rebuilds from `pip install geosave` alone: every component and
sub-processor it names must be one the library registers. To deploy a model,
add its components and sub-processors to geosave first.

    processor ──ChainModel.from_processor──▶ save_pretrained / push_to_hub
        ──▶ config.json + model.safetensors ──load_processor──▶ processor
"""

from __future__ import annotations

import warnings
from typing import Any, TypedDict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from geosave.__about__ import __version__
from geosave.ml.chain import ModelChain
from geosave.ml.components.build import build_model
from geosave.ml.components.registry import COMPONENTS
from geosave.ml.factory import BuildSpec, Registry
from geosave.ml.processing.processor import Processor, ProcessorSpec, build_processor
from geosave.ml.processing.roles import (
    INGESTORS,
    POSTPROCESSORS,
    PREDICTORS,
    PREPROCESSORS,
)

ARTIFACT_FORMAT_VERSION = 2


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


class ChainConfig(PretrainedConfig):
    """Record a native chain, and the processor that runs it, in a release.

    Args:
        stages: Ordered stage names and construction specifications. A list,
            because Transformers writes `config.json` with sorted keys.
        processor: Name and arguments of each sub-processor of the processor
            that runs the model; None when the release carries none.
        format_version: GeoSave release format version.
        geosave_version: GeoSave version that saved the release.
        **kwargs: Standard Transformers configuration fields.

    Raises:
        ValueError: The format version is unsupported or stage names repeat.

    Warns:
        UserWarning: The release was saved by another GeoSave version. Its
            weights still have to match the rebuilt chain exactly.
    """

    model_type = "geosave_chain"

    def __init__(
        self,
        stages: list[StageConfig] | None = None,
        processor: dict[str, Any] | None = None,
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
            warnings.warn(
                f"release saved by GeoSave {geosave_version}; "
                f"installed version is {__version__}",
                stacklevel=2,
            )
        super().__init__(**kwargs)
        self.stages = stages if stages is not None else []
        self.processor = processor
        self.format_version = format_version
        self.geosave_version = geosave_version
        names = [entry["stage"] for entry in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Each stage must have a unique name")


class ChainModel(PreTrainedModel):
    """Load, save, and push a native model chain as a Transformers model.

    Args:
        config: Ordered construction specifications.
        chain: Existing chain to wrap without rebuilding or copying weights.
            Its specifications must match config. Omit when loading a release.

    Raises:
        ValueError: The supplied chain does not match the configuration.

    Examples:
        >>> release = ChainModel.from_processor(processor)
        >>> commit = release.push_to_hub("org/model")
        >>> processor = load_processor("org/model", revision=commit.oid)
    """

    config_class = ChainConfig
    base_model_prefix = "chain"

    def __init__(
        self,
        config: ChainConfig,
        *,
        chain: ModelChain | None = None,
    ) -> None:
        super().__init__(config)
        specs = {entry["stage"]: entry["spec"] for entry in config.stages}
        if chain is not None and list(chain.stage_specs.items()) != list(specs.items()):
            raise ValueError("Chain construction specifications do not match config")
        self.chain = chain if chain is not None else build_model(specs)
        self.add_model_tags("geosave")
        self.post_init()

    @classmethod
    def from_chain(cls, chain: ModelChain) -> ChainModel:
        """Wrap a trained chain for release without a processor, sharing its weights.

        Args:
            chain: Model built from library components by registered name.

        Returns:
            Transformers model sharing the chain and its training mode.

        Raises:
            ValueError: The chain has no construction specifications, or a
                stage is not a library component.
        """
        return cls.wrap(chain, None)

    @classmethod
    def from_processor(cls, processor: Processor) -> ChainModel:
        """Wrap a processor's chain for release with the processor's spec.

        Args:
            processor: Processor built by `build_processor` around a chain,
                from library sub-processors by registered name.

        Returns:
            Transformers model sharing the chain; `load_processor` rebuilds
            the processor around it.

        Raises:
            TypeError: The processor's model is not a `ModelChain`.
            ValueError: The processor has no spec, or a stage or
                sub-processor is not part of the library.
        """
        if not isinstance(processor.model, ModelChain):
            raise TypeError("the processor's model must be a ModelChain")
        if processor.spec is None:
            raise ValueError(
                "the processor has no spec; build it with `build_processor`"
            )

        roles = {
            "ingestor": (INGESTORS, processor.spec.ingestor),
            "preprocessor": (PREPROCESSORS, processor.spec.preprocessor),
            "predictor": (PREDICTORS, processor.spec.predictor),
            "postprocessor": (POSTPROCESSORS, processor.spec.postprocessor),
        }
        for role, (registry, spec) in roles.items():
            check_library_spec(registry, spec, role)
        return cls.wrap(processor.model, processor.spec)

    @classmethod
    def wrap(cls, chain: ModelChain, processor: ProcessorSpec | None) -> ChainModel:
        """Wrap a chain and an optional processor spec, sharing the chain's weights.

        Raises:
            ValueError: The chain has no construction specifications, or a
                stage is not a library component.
        """
        stages = []
        for stage, stage_spec in chain.stage_specs.items():
            registry = COMPONENTS.get(stage, Registry(nn.Module))
            check_library_spec(registry, BuildSpec.model_validate(stage_spec), stage)
            stages.append(StageConfig(stage=stage, spec=stage_spec))

        config = ChainConfig(
            stages=stages,
            processor=None
            if processor is None
            else processor.model_dump(exclude_none=True),
        )
        model = cls(config, chain=chain)
        model.training = chain.training
        return model

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
                "ChainModel requires an exact checkpoint match. " + "; ".join(problems)
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


def check_library_spec(registry: Registry[Any], spec: BuildSpec, label: str) -> None:
    """Refuse a spec that `pip install geosave` alone cannot rebuild.

    Args:
        registry: Registry the spec selects from.
        spec: Component or sub-processor specification.
        label: Stage or role named in the error.

    Raises:
        ValueError: The spec selects by class path, or names a constructor
            defined outside geosave, such as one a project registers.
    """
    if spec.name is None:
        raise ValueError(
            f"{label} is selected by class path {spec.class_path!r}; add it to "
            f"geosave and register it to release it"
        )

    factory = registry.resolve(spec)
    if not factory.__module__.startswith("geosave."):
        raise ValueError(
            f"{label} {spec.name!r} is defined in {factory.__module__}, outside "
            f"geosave; add it to the library to release it"
        )


def load_processor(repo: str, **kwargs: Any) -> Processor:
    """Load a released model with the processor that runs it.

    Args:
        repo: Hub repository id or local folder of a release.
        **kwargs: Passed to `ChainModel.from_pretrained`, such as `revision`.

    Returns:
        The release's processor holding the model in evaluation mode.

    Raises:
        ValueError: The release carries no processor.
        KeyError: A sub-processor name is not registered.

    Examples:
        >>> processor = load_processor("org/dynamic-world")
        >>> processor.predict_path("scene.tif")
        <xarray.Dataset> ...
    """
    model = ChainModel.from_pretrained(repo, **kwargs)
    if model.config.processor is None:
        raise ValueError(f"the release {repo!r} carries no processor")

    model.eval()
    return build_processor(ProcessorSpec.model_validate(model.config.processor), model)


AutoConfig.register(ChainConfig.model_type, ChainConfig)
AutoModel.register(ChainConfig, ChainModel)
