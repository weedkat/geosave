"""Persist native model chains and their pipelines through the Transformers file contract.

A release's chain rebuilds from `pip install geosave` alone: every component
it names must be one the library registers. A sub-processor is either a
library one, by name, or a project's own, by class path; the second kind
rebuilds wherever the project's code is installed, such as a server image
built from the workspace.

    pipeline ──ChainModel.from_pipeline──▶ save_pretrained / push_to_hub
        ──▶ config.json + model.safetensors ──┬─ AutoModel.from_pretrained ──▶ model
                                              └─ load_pipeline ──▶ pipeline
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
from geosave.ml.processing.raster.ingestion.base import INGESTORS
from geosave.ml.processing.raster.postprocessing.base import POSTPROCESSORS
from geosave.ml.processing.raster.prediction.base import PREDICTORS
from geosave.ml.processing.raster.preprocessing.base import PREPROCESSORS
from geosave.ml.processing.raster.pipeline import RasterPipeline

ARTIFACT_FORMAT_VERSION = 4


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
    """Record a native chain, and the pipeline that runs it, in a release.

    Each sub-processor is a `BuildSpec` as JSON: its registered name or
    class path, with every constructor argument.

    Args:
        stages: Ordered stage names and construction specifications. A list,
            because Transformers writes `config.json` with sorted keys.
        preprocessor: The pipeline's preprocessor; None when the release
            carries the chain alone.
        predictor: The pipeline's predictor.
        postprocessor: The pipeline's postprocessor.
        ingestor: The pipeline's ingestor; None when it has none.
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
        preprocessor: dict[str, Any] | None = None,
        predictor: dict[str, Any] | None = None,
        postprocessor: dict[str, Any] | None = None,
        ingestor: dict[str, Any] | None = None,
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
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.ingestor = ingestor
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
        >>> release = ChainModel.from_pipeline(pipeline)
        >>> commit = release.push_to_hub("org/model")
        >>> model = AutoModel.from_pretrained("org/model", revision=commit.oid)
        >>> pipeline = load_pipeline("org/model", revision=commit.oid)
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
        """Wrap a trained chain for release without a pipeline, sharing its weights.

        Args:
            chain: Model built from library components by registered name.

        Returns:
            Transformers model sharing the chain and its training mode.

        Raises:
            ValueError: The chain has no construction specifications, or a
                stage is not a library component.
        """
        config = ChainConfig(stages=read_stages(chain))
        return cls.wrap(chain, config)

    @classmethod
    def from_pipeline(cls, pipeline: RasterPipeline) -> ChainModel:
        """Wrap a pipeline's chain for release with its sub-processors.

        Each sub-processor's class and constructor arguments are read from
        the sub-processor itself, so the pipeline that was run is the one
        released.

        Args:
            pipeline: Pipeline around a chain. A library sub-processor is
                stored by registered name, a project's own by class path.

        Returns:
            Transformers model sharing the chain; `load_pipeline` rebuilds
            the pipeline around it.

        Raises:
            TypeError: The pipeline's model is not a `ModelChain`, or a
                sub-processor keeps no attribute for a constructor argument.
            ValueError: A stage is not part of the library, or a project's
                sub-processor cannot be imported by its class path, as a
                class defined inside a function.
        """
        if not isinstance(pipeline.model, ModelChain):
            raise TypeError("the pipeline's model must be a ModelChain")

        ingestor = None
        if pipeline.ingestor is not None:
            ingestor = read_deployable_spec(INGESTORS, pipeline.ingestor)

        config = ChainConfig(
            stages=read_stages(pipeline.model),
            preprocessor=read_deployable_spec(PREPROCESSORS, pipeline.preprocessor),
            predictor=read_deployable_spec(PREDICTORS, pipeline.predictor),
            postprocessor=read_deployable_spec(POSTPROCESSORS, pipeline.postprocessor),
            ingestor=ingestor,
        )
        return cls.wrap(pipeline.model, config)

    @classmethod
    def wrap(cls, chain: ModelChain, config: ChainConfig) -> ChainModel:
        """Wrap a chain under its release configuration, sharing the chain's weights."""
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


def read_stages(chain: ModelChain) -> list[StageConfig]:
    """Return a chain's stages as a release stores them, in order.

    Raises:
        ValueError: The chain has no construction specifications, or a
            stage is not a library component.
    """
    stages = []
    for stage, stage_spec in chain.stage_specs.items():
        registry = COMPONENTS.get(stage, Registry(nn.Module))
        check_library_spec(registry, BuildSpec.model_validate(stage_spec), stage)
        stages.append(StageConfig(stage=stage, spec=stage_spec))
    return stages


def read_deployable_spec(registry: Registry[Any], built: object) -> dict[str, Any]:
    """Return a sub-processor's spec as another machine can rebuild it.

    A library sub-processor rebuilds by its registered name. A project's own
    rebuilds by class path wherever the project's code is installed, as in
    the image a server is built from; a name the project registered would be
    unknown there until the project's module is imported.

    Args:
        registry: Registry of the sub-processor's role.
        built: The sub-processor.

    Returns:
        Its `BuildSpec` as JSON-ready values.

    Raises:
        TypeError: The sub-processor is not of the registry's role, or keeps
            no attribute for a constructor argument.
        ValueError: Its class cannot be imported by its class path.
    """
    spec = registry.read_spec(built)

    cls = type(built)
    if spec.name is not None and not cls.__module__.startswith("geosave."):
        class_path = f"{cls.__module__}.{cls.__qualname__}"
        spec = BuildSpec(class_path=class_path, init_args=spec.init_args)

    # A release that could not be loaded again is refused when it is made.
    try:
        registry.resolve(spec)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"{cls.__qualname__} cannot be imported by its class path "
            f"{spec.class_path!r}; define it at module level to release it"
        ) from error
    return spec.model_dump(exclude_none=True)


def load_pipeline(repo: str, **kwargs: Any) -> RasterPipeline:
    """Load a released model with the pipeline that runs it.

    `AutoModel.from_pretrained` loads the model alone.

    Args:
        repo: Hub repository id or local folder of a release.
        **kwargs: Passed to `ChainModel.from_pretrained`, such as `revision`.

    Returns:
        The release's pipeline; its `model` is the native chain in
        evaluation mode.

    Raises:
        ValueError: The release carries no pipeline.
        KeyError: A sub-processor name is not registered.
        ModuleNotFoundError: A sub-processor selected by class path belongs
            to a project whose code is not installed here.

    Examples:
        >>> pipeline = load_pipeline("org/dynamic-world")
        >>> pipeline.predict(rasters)
        <xarray.Dataset> ...
    """
    release = ChainModel.from_pretrained(repo, **kwargs)
    config = release.config
    if config.preprocessor is None:
        raise ValueError(f"the release {repo!r} carries no pipeline")

    release.eval()
    ingestor = None
    if config.ingestor is not None:
        ingestor = INGESTORS.build(BuildSpec.model_validate(config.ingestor))

    return RasterPipeline(
        release.chain,
        preprocessor=PREPROCESSORS.build(BuildSpec.model_validate(config.preprocessor)),
        predictor=PREDICTORS.build(BuildSpec.model_validate(config.predictor)),
        postprocessor=POSTPROCESSORS.build(
            BuildSpec.model_validate(config.postprocessor)
        ),
        ingestor=ingestor,
    )


AutoConfig.register(ChainConfig.model_type, ChainConfig)
AutoModel.register(ChainConfig, ChainModel)
