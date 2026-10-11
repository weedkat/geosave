"""Portable model declarations and strict YAML persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator
import xarray as xr
import yaml
from upath import UPath
from yaml.nodes import ScalarNode

from .base import Name, SpecModel, Text
from .call import CallSpec, Ref
from .cuts import FramesSpec, ChipsSpec
from .rasters import RasterRequirement
from .stage import StageSpec
from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.io import storage
from geosave_engine.geodata.io.storage import Location

if TYPE_CHECKING:
    import geopandas as gpd


class _Loader(yaml.SafeLoader):
    """Read ordinary mappings strictly and retain inert references."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key, value in self.construct_pairs(node, deep=deep):
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key {key!r}")
            result[key] = value
        return result


class _Dumper(yaml.SafeDumper):
    """Write references as tags and literals through the safe dumper."""

    def represent_data(self, data):
        if isinstance(data, Ref):
            return self.represent_scalar("!ref", data.path)
        return super().represent_data(data)


def _construct_ref(loader: _Loader, node: ScalarNode) -> Ref:
    """Construct one inert scalar reference."""
    return Ref(loader.construct_scalar(node))


_Loader.add_constructor("!ref", _construct_ref)


class TransformSpec(SpecModel):
    """Declare one tensor transform by its Kornia class name.

    Args:
        name: Class in `kornia.augmentation`, such as `Normalize`.
        init_args: Its constructor arguments.
    """

    name: Text
    init_args: dict[str, JsonValue] = Field(default_factory=dict)


class ModelSpec(SpecModel):
    """Describe model raster requirements and inert processing declarations.

    Args:
        schema_version: Document format version.
        rasters: Rasters the model reads, by name.
        preprocessing: Named raster calls run on each sample.
        frames: How a sample's time axis is cut. None cuts nothing.
        chips: How a raster is cut in space and merged back.
        inputs: What the model chain takes, by input name. A reference is a
            raster's pixels.
        context: Optional model context recipe reading the same rasters as
            `inputs`, shared by training and inference.
        transforms: Tensor transforms per pixel input, run after augmentation.

    Raises:
        ValueError: Preprocessing reads an undeclared raster, an input reads
            neither a preprocessing result nor a raster, or transforms name
            an input that is not pixels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[2]
    rasters: dict[Name, RasterRequirement]
    preprocessing: StageSpec = Field(default_factory=StageSpec)
    frames: FramesSpec | None = None
    chips: ChipsSpec | None = None
    inputs: dict[Name, Ref] = Field(default_factory=dict)
    context: CallSpec | None = None
    transforms: dict[Name, list[TransformSpec]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_preprocessing_inputs(self) -> Self:
        unknown = set(self.preprocessing.external_inputs) - self.rasters.keys()
        if unknown:
            raise ValueError(
                f"Preprocessing inputs must be declared rasters: {sorted(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_inputs(self) -> Self:
        known = self.rasters.keys() | set(self.preprocessing)
        if unknown := set(self.input_rasters) - known:
            raise ValueError(
                "Model inputs reference neither a preprocessing result nor a "
                f"raster: {sorted(unknown)}"
            )
        if self.context is not None and (unknown := self.context.inputs - known):
            raise ValueError(
                "Context references neither a preprocessing result nor a "
                f"raster: {sorted(unknown)}"
            )
        if unknown := self.transforms.keys() - set(self.pixel_inputs):
            raise ValueError(
                f"Transforms name inputs that are not pixels: {sorted(unknown)}"
            )
        return self

    @property
    def pixel_inputs(self) -> tuple[str, ...]:
        """Return the model inputs that are a raster's pixels."""
        return tuple(name for name in self.inputs)

    @property
    def input_rasters(self) -> tuple[str, ...]:
        """Return the results and rasters model inputs read, in first-use order."""
        return tuple(dict.fromkeys(value.root for value in self.inputs.values()))

    def load_rasters(self, anchor: GeoAnchor, /) -> dict[str, xr.Dataset]:
        """Load every declared raster from STAC onto an anchor.

        Not settled yet: read rasters from a catalog through `cuts.stacks`
        and `row.gs.to_stack()` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ModelSpec.load_rasters is not settled; read rasters from a catalog "
            "with cuts.stacks and row.gs.to_stack()"
        )

    def preprocess(self, rasters: Mapping[str, Any], /) -> dict[str, Any]:
        """Select the declared rasters and run preprocessing on them.

        Not settled yet: redesigned together with frames, chips, inputs and
        context.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ModelSpec.preprocess is not settled")

    def prepare(
        self,
        raw: gpd.GeoDataFrame,
        output: Location,
        *,
        on: str,
        keep: Sequence[str] = (),
    ) -> UPath:
        """Preprocess every window of a raw catalog and save what the model reads.

        Not settled yet: until the notebook examples fix the flow, save model
        inputs with `to_cog` and catalogue them with `stac.create_items`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ModelSpec.prepare is not settled; save the model inputs with "
            "to_cog and catalogue them with stac.create_items"
        )

    def model_context(self, rasters: Mapping[str, Any]) -> dict[str, Any]:
        """Encode the sample's rasters with the declared context recipe.

        Not settled yet: redesigned together with frames, chips, inputs and
        context.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ModelSpec.model_context is not settled")

    def model_inputs(
        self,
        rasters: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine the declared raster inputs with their context.

        Not settled yet: redesigned together with frames, chips, inputs and
        context.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ModelSpec.model_inputs is not settled")

    def save(self, path: Location) -> UPath:
        """Write this spec as YAML, with references as `!ref` tags.

        Args:
            path: YAML file to write, as a path, URL, or UPath.

        Returns:
            The written file.

        Examples:
            >>> spec.save("configs/model_spec.yaml")
            PosixUPath('configs/model_spec.yaml')
        """
        target = storage.location(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.dump(self.model_dump(), Dumper=_Dumper, sort_keys=False)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Location) -> Self:
        """Read a model_spec.yaml without importing the calls it declares.

        Args:
            path: YAML file, as a path, URL, or UPath.

        Returns:
            The spec the file declares.

        Examples:
            >>> ModelSpec.load("configs/model_spec.yaml").inputs
            {'image': Ref('image')}
        """
        payload = storage.location(path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.load(payload, Loader=_Loader))
