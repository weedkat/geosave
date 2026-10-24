"""Portable model declarations and strict YAML persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator
import xarray as xr
import yaml
from upath import UPath
from yaml.nodes import ScalarNode

from .base import Name, SpecModel, Text
from .call import CallSpec, Ref
from .windows import FramesSpec, ChipsSpec
from .rasters import RasterRequirement
from .stage import StageSpec
from geosave.geodata import stac, tree
from geosave.geodata.core import GeoAnchor
from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location
from geosave.geodata.stac.create import ID_FORMAT

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core import GeoCatalog


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

    Attributes:
        name: Class in `kornia.augmentation`, such as `Normalize`.
        init_args: Its constructor arguments.
    """

    name: Text
    init_args: dict[str, JsonValue] = Field(default_factory=dict)


class ModelSpec(SpecModel):
    """Describe model raster requirements and inert processing declarations.

    Attributes:
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
        # Context is encoded from what the model inputs read, so it reads no more.
        if self.context is not None and (
            unknown := self.context.inputs - set(self.input_rasters)
        ):
            raise ValueError(
                f"Context references what no model input reads: {sorted(unknown)}"
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
        """Load and select every declared raster from its STAC recipe.

        Args:
            anchor: Grid and time the rasters are loaded on.

        Returns:
            Raster name mapped to its selected lazy raster.

        Raises:
            ValueError: A raster declares no `stac` block, or a loaded raster
                misses its requirement.

        Examples:
            >>> spec.load_rasters(anchor)["sentinel_2_l2a"].gs.variables
            ('B02', 'B03', 'B04', 'B08')
        """
        if missing := [n for n, r in self.rasters.items() if r.stac is None]:
            raise ValueError(f"Rasters declare no `stac` block: {missing}")
        rasters = {}
        for name, requirement in self.rasters.items():
            assert requirement.stac is not None
            try:
                rasters[name] = requirement.select_raster(
                    requirement.stac.load_raster(anchor)
                )
            except Exception as error:
                error.add_note(f"While loading raster {name!r}")
                raise
        return rasters

    def preprocess(self, rasters: Mapping[str, Any], /) -> dict[str, Any]:
        """Select the declared rasters and run preprocessing on them.

        Args:
            rasters: Rasters by name; names this spec does not declare, such
                as labels, are ignored.

        Returns:
            Each selected raster and each preprocessing result, by name.

        Raises:
            KeyError: A declared raster is not supplied, or lacks a declared
                variable or coordinate.
            ValueError: A raster misses its requirement.

        Examples:
            >>> sorted(spec.preprocess({"sentinel_2_l2a": scene, "label": label}))
            ['image', 'sentinel_2_l2a']
        """
        if missing := [name for name in self.rasters if name not in rasters]:
            raise KeyError(f"Missing declared rasters: {missing}")
        selected = {}
        for name, requirement in self.rasters.items():
            try:
                selected[name] = requirement.select_raster(rasters[name])
            except Exception as error:
                error.add_note(f"While selecting raster {name!r}")
                raise
        return {**selected, **self.preprocessing.run(selected)}

    def ingest(
        self,
        anchor: GeoAnchor,
        path: Location | Mapping[str, Location],
        *,
        id_format: str = ID_FORMAT,
    ) -> GeoDataFrame[GeoCatalog]:
        """Load every declared raster on an anchor, save it, and list it as Items.

        Training and inference ingest alike and get the same table. Pixels are
        saved as the source publishes them; `preprocess` runs when they are
        read. Each raster is saved as the collection its spec name says, so
        `row.gs.to_xarray().gs.rasters` hands `preprocess` the names it reads.

        Args:
            anchor: Grid and time every raster is loaded on.
            path: COG path template, as `DataTree.gs.to_cog` takes it, such as
                `"data/{collection}/{time:%Y%m%d}_{band}.tif"`.
            id_format: `GeoAnchor.format` template each Item's id is filled
                from, as `stac.create_items` takes it.

        Returns:
            Item table, one Item per raster and instant.

        Raises:
            ValueError: A raster declares no `stac` block, or misses its
                requirement.
            FileExistsError: A file the template names exists.

        Examples:
            >>> items = spec.ingest(anchor, "data/s0/{collection}/{band}.tif")
            >>> items.gs.sample().iloc[0].gs.to_xarray().gs.groups
            ('sentinel_2_l2a',)
        """
        scenes = tree(self.load_rasters(anchor)).gs.to_cog(path)
        return stac.create_catalog(stac.create_items(scenes, id_format=id_format))

    def model_context(self, rasters: Mapping[str, Any]) -> dict[str, Any]:
        """Encode a sample's rasters with the declared context recipe.

        Args:
            rasters: Prepared rasters, as `model_inputs` takes them.

        Returns:
            What the recipe returns, or an empty dict without one.

        Raises:
            KeyError: A raster the recipe reads is absent.
            TypeError: The recipe returns something other than a dict.
        """
        if self.context is None:
            return {}
        values = self.context.invoke(self.context.select_inputs(rasters))
        if not isinstance(values, dict):
            raise TypeError(
                f"model context must return a dict, got {type(values).__name__}"
            )
        return values

    def model_inputs(
        self,
        rasters: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bind each input to its prepared raster and add their context.

        Args:
            rasters: Prepared rasters, as `row.gs.to_xarray().gs.rasters`
                reads them.
            context: Cached encoding of these rasters. None encodes them.

        Returns:
            Each input's raster and each context value, by name.

        Raises:
            KeyError: A raster an input reads is absent.
            ValueError: Context would replace a pixel input.

        Examples:
            >>> sorted(spec.model_inputs(chip.gs.rasters))
            ['image', 'location_coords', 'temporal_coords']
        """
        pixels = {name: ref.resolve(rasters) for name, ref in self.inputs.items()}
        encoded = self.model_context(rasters) if context is None else dict(context)
        if overlap := pixels.keys() & encoded.keys():
            raise ValueError(f"Context cannot replace pixel inputs: {sorted(overlap)}")
        return {**pixels, **encoded}

    def save(self, path: Location) -> UPath:
        """Write this spec as YAML, with references as `!ref` tags.

        Only declared values are written; defaults are left to loading.

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
        # Defaults are left out, so the file reads as the spec was written.
        declared = self.model_dump(exclude_defaults=True)
        payload = yaml.dump(declared, Dumper=_Dumper, sort_keys=False)
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
