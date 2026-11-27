"""Run a model over a region: ingest, preprocess, predict, postprocess."""

import copy
import inspect
from collections.abc import Collection, Mapping
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Self

import xarray as xr
import yaml
from huggingface_hub import ModelHubMixin, hf_hub_download
from jsonargparse import ArgumentParser

from geosave.geodata import GeoRegion, open_raster_tree
from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location

from .ingestion.base import IngestOutput, Ingestor
from .postprocessing.base import PostprocessOutput, Postprocessor
from .prediction.base import Predictor
from .preprocessing.preprocessor import Preprocessor

_CONFIG_FILE = "pipeline.yaml"


class Pipeline(ModelHubMixin):
    """Turn a region into one task's product with one model.

    Each stage is identified by one thing, and hands the next stage one
    output whose fields every implementation of that stage fills:

    - ingestors: the source collections they read, one subtree each;
      together they give an `IngestOutput`
    - preprocessor: the model inputs it makes, one per temporal window;
      a `PreprocessOutput`
    - predictor: the ML framework it runs; a `PredictOutput`
    - postprocessor: the task, which decides the product: a raster
      Dataset for dense tasks, a GeoDataFrame for detections; a
      `PostprocessOutput`. jsonargparse builds no subscripted generic, so
      the product is typed `Any`.

    Without a store, every stage works in memory. With a store, a folder,
    the run stages what is too large for memory there as Zarr:
    `sources.zarr` holds every collection, and the postprocessor stages its
    product under `product/`. A store is internal to one run, not a
    publication.

    Args:
        ingestors: Readers of the source collections.
        preprocessor: Maker of the model inputs.
        predictor: Runner of the model.
        postprocessor: Maker of the product from the model outputs.

    Examples:
        >>> pipeline = Pipeline.from_pretrained("org/landcover-s2")
        >>> classes = pipeline(region).product
        >>> classes = pipeline(region, store="runs/kenya").product
        >>> pipeline.push_to_hub("org/landcover-s2")
    """

    def __init__(
        self,
        ingestors: list[Ingestor],
        preprocessor: Preprocessor,
        predictor: Predictor,
        postprocessor: Postprocessor,
    ) -> None:
        self.ingestors = ingestors
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor

    @classmethod
    def from_config(cls, config: Location | Mapping[str, Any], model: Any) -> Self:
        """Build a pipeline from its configuration and the model its predictor runs.

        Each stage is selected by `class_path` with its `init_args`, and every
        value is checked against the constructors' type hints before anything
        is built. The configuration leaves out the model; it is passed to the
        predictor as its `model` argument.

        Args:
            config: The configuration itself, or the location of its YAML file.
            model: The model the predictor runs.

        Returns:
            The pipeline the configuration describes.

        Raises:
            jsonargparse.ArgumentError: A stage or a value does not fit its type.

        Examples:
            >>> model = ModelChain.from_pretrained("org/landcover-s2")
            >>> pipeline = Pipeline.from_config("pipeline.yaml", model)
        """
        if isinstance(config, Mapping):
            stages = config
        else:
            stages = yaml.safe_load(storage.create_upath(config).read_text())

        parser = ArgumentParser(exit_on_error=False)
        parser.add_class_arguments(cls)
        parsed = parser.parse_object(_place_model(stages, model))
        return cls(**parser.instantiate_classes(parsed))

    def to_config(self) -> dict[str, Any]:
        """Return the configuration `from_config` rebuilds this pipeline from.

        Each stage's constructor arguments are read from the attributes of the
        same name, as they are now, so a value changed after construction is
        the one returned. The predictor's model is left out.

        Returns:
            The stages as jsonargparse reads them, each `class_path` with its
            `init_args`.

        Raises:
            AttributeError: An object keeps no attribute for a constructor
                argument: a stage breaking the rule, or a value a config
                cannot hold, such as an array or a function.
            ValueError: A stage's class is defined in `__main__`, such as in a
                notebook, so no other session can import it.
            jsonargparse.ArgumentError: A value does not fit its constructor's
                type hint.

        Examples:
            >>> pipeline.to_config()["predictor"]
            {'class_path': 'geosave.ml.torch.prediction.predictor.TorchPredictor',
             'init_args': {'chip_size': 224, ...}}
        """
        config = {
            "ingestors": _read_constructor_config(self.ingestors),
            "preprocessor": _read_constructor_config(self.preprocessor),
            "predictor": _read_constructor_config(self.predictor, skip={"model"}),
            "postprocessor": _read_constructor_config(self.postprocessor),
        }

        # Parse it as `from_config` will, so a value no config can hold is
        # refused now rather than when the release is loaded.
        parser = ArgumentParser(exit_on_error=False)
        parser.add_class_arguments(type(self))
        parser.parse_object(_place_model(config, self.predictor.model))
        return config

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write `pipeline.yaml`, then the model's own files beside it."""
        model_class = type(self.predictor.model)
        config = {
            "model": f"{model_class.__module__}.{model_class.__qualname__}",
            **self.to_config(),
        }
        text = yaml.safe_dump(config, sort_keys=False)
        (save_directory / _CONFIG_FILE).write_text(text)
        self.predictor.model.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs: Any,
    ) -> Self:
        """Load the model with its class's `from_pretrained`, then build the stages around it.

        Keyword arguments `from_pretrained` does not take itself go to the
        model's `from_pretrained`.
        """
        path = Path(model_id) / _CONFIG_FILE
        if not path.is_file():
            path = hf_hub_download(
                model_id,
                _CONFIG_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )
        config = yaml.safe_load(Path(path).read_text())

        module_name, _, class_name = config.pop("model").rpartition(".")
        model_class = getattr(import_module(module_name), class_name)
        model = model_class.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )
        return cls.from_config(config, model)

    def __call__(
        self, region: GeoRegion, store: Location | None = None
    ) -> PostprocessOutput[Any]:
        """Return the product over a region, one temporal window after another.

        Args:
            region: Place and time span to predict.
            store: Folder this run stages its sources and product in; None
                holds everything in memory.

        Returns:
            The postprocess stage's output: the product over every window,
            lazy over the store when one is given.

        Raises:
            ValueError: The region has no complete temporal window.
            FileExistsError: The store already holds a run.

        Examples:
            >>> classes = pipeline(region, store="runs/kenya").product
            >>> classes.class_id.dims
            ('time', 'y', 'x')
        """
        folder = None if store is None else storage.create_upath(store)
        ingested = self.ingest(region, folder)
        preprocessed = self.preprocessor(ingested)
        predicted = self.predictor(preprocessed)
        staged = None if folder is None else folder / "product"
        return self.postprocessor(predicted, staged)

    def ingest(self, region: GeoRegion, store: Location | None = None) -> IngestOutput:
        """Read every collection over a region into one tree, in memory or staged.

        Each ingestor's subtree becomes a child of the root under its own name.
        Staged, every collection is written to `<store>/sources.zarr` in one
        computation and read back lazily from there.

        Args:
            region: Place and time span to read.
            store: Folder to stage the sources in; None loads every pixel
                into memory.

        Returns:
            The ingest stage's output: one tree with one child per collection.

        Raises:
            ValueError: Two ingestors return the same node name.
            FileExistsError: The store already holds `sources.zarr`.

        Examples:
            >>> ingested = pipeline.ingest(region)
            >>> list(ingested.sources.children)
            ['s2', 's1']
        """
        children = {}
        for ingestor in self.ingestors:
            for name, child in ingestor(region).children.items():
                if name in children:
                    raise ValueError(f"two ingestors return the node {name!r}")
                children[name] = child.copy()
        sources = xr.DataTree(children=children)

        if store is None:
            return IngestOutput(sources.compute())
        path = storage.create_upath(store) / "sources.zarr"
        sources.gs.to_zarr(path)
        return IngestOutput(open_raster_tree(path, chunks={}))


def _place_model(config: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Return a copy of a pipeline configuration with the model as the predictor's `model` argument.

    The copy is deep because jsonargparse converts the values it parses in
    place, such as an enum member's name into the member.
    """
    placed = copy.deepcopy(dict(config))
    placed["predictor"].setdefault("init_args", {})["model"] = model
    return placed


def _read_constructor_config(value: Any, skip: Collection[str] = ()) -> Any:
    """Return the configuration jsonargparse rebuilds a value from.

    A plain value is returned as it is, an enum member as its name, and a
    list or dict entry by entry. Any other object is returned as its class
    and its constructor arguments, each read from the attribute of the same
    name:

        {"class_path": "geosave.ml.pipeline.preprocessing.steps.Composite",
         "init_args": {"freq": "MS", "method": "median", ...}}

    Args:
        value: A stage, or one of its constructor arguments.
        skip: Constructor arguments of `value` to leave out.

    Raises:
        AttributeError: An object keeps no attribute for a constructor argument.
        ValueError: An object's class is defined in `__main__`.
    """
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        items = []
        for item in value:
            items.append(_read_constructor_config(item))
        return items
    if isinstance(value, dict):
        entries = {}
        for key, entry in value.items():
            entries[key] = _read_constructor_config(entry)
        return entries

    cls = type(value)
    if cls.__module__ == "__main__":
        raise ValueError(
            f"{cls.__qualname__} is defined in __main__, so no other session can "
            f"import it; move it into an importable module"
        )

    init_args = {}
    for name, parameter in inspect.signature(cls).parameters.items():
        # `object.__init__` gives a class without its own constructor `*args, **kwargs`.
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in skip:
            continue
        init_args[name] = _read_constructor_config(getattr(value, name))
    return {
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
        "init_args": init_args,
    }
