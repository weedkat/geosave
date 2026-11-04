"""Portable model declarations and strict YAML persistence."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import xarray as xr
import yaml
from pydantic import ConfigDict, Field, model_validator
from upath import UPath
from yaml.nodes import ScalarNode

from odc.geo.geobox import GeoBox
from geosave.geodata.core import GeoAnchor
from geosave.geodata.stac.catalog import from_items
from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location
from geosave.geodata.conventions import ITEM_ID_FORMAT

from .base import Name, SpecModel
from .call import Ref
from .rasters import RasterRequirement
from .stage import StageSpec
from .tensors import TensorRequirement

if TYPE_CHECKING:
    import pystac

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core import GeoCatalog
    from geosave.geodata.transform.warp import Target


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


class ModelSpec(SpecModel):
    """Execute a YAML model-processing recipe on native values.

    Attributes:
        schema_version: Recipe version, currently 3.
        rasters: Raw raster selection and metadata requirements.
        preprocessing: Calls on a selected sample with sufficient spatial/time support.
        encoding: Calls producing tensors, context, and per-sample inverse state.
        inputs: Encoded values bound to model argument names.
        pixels: Input names that training may augment as images.
        tensors: Optional sample-shape and dtype contracts for model inputs.
        transforms: Calls on named inputs, shared after augmentation and at inference.
        decoding: Per-sample output calls with prediction and that sample's state.
        postprocessing: Calls on a merged prediction, such as final class selection.

    Calls run only when their stage is invoked. The caller owns acquisition,
    sampling, chip layout, augmentation, model execution, and merging. No stage
    is automatically moved across those boundaries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[3]
    rasters: dict[Name, RasterRequirement]
    preprocessing: StageSpec = Field(default_factory=StageSpec)
    encoding: StageSpec = Field(default_factory=StageSpec)
    inputs: dict[Name, Ref] = Field(default_factory=dict)
    pixels: tuple[Name, ...] = ()
    tensors: dict[Name, TensorRequirement] = Field(default_factory=dict)
    transforms: StageSpec = Field(default_factory=StageSpec)
    decoding: StageSpec = Field(default_factory=StageSpec)
    postprocessing: StageSpec = Field(default_factory=StageSpec)

    @model_validator(mode="after")
    def _validate_stages(self) -> Self:
        known = set(self.rasters)
        self.preprocessing.validate_inputs(known)
        known.update(self.preprocessing)
        self.encoding.validate_inputs(known)
        known.update(self.encoding)
        if "prediction" in known:
            raise ValueError("prediction is reserved for model outputs")
        unknown = {ref.root for ref in self.inputs.values()} - known
        if unknown:
            raise ValueError(
                f"Model inputs reference unavailable values: {sorted(unknown)}"
            )
        if len(set(self.pixels)) != len(self.pixels):
            raise ValueError("pixels must name each input once")
        unknown = (set(self.pixels) | self.tensors.keys()) - self.inputs.keys()
        if unknown:
            raise ValueError(
                f"Input requirements name undeclared inputs: {sorted(unknown)}"
            )
        self.transforms.validate_inputs(self.inputs.keys())
        self.decoding.validate_inputs(known | {"prediction"})
        self.postprocessing.validate_inputs({"prediction"})
        return self

    @property
    def input_rasters(self) -> tuple[str, ...]:
        """Return prepared roots needed for encoding and decoding, in first-use order."""
        known = self.rasters.keys() | set(self.preprocessing)
        names = dict.fromkeys(self.encoding.external_inputs)
        for ref in self.inputs.values():
            if ref.root in known:
                names.setdefault(ref.root, None)
        for name in self.decoding.external_inputs:
            if name in known and name not in self.encoding:
                names.setdefault(name, None)
        return tuple(names)

    def load_rasters(
        self, anchor: GeoAnchor, /, *, geobox: GeoBox
    ) -> dict[str, xr.Dataset]:
        """Load and select every declared raster from its STAC recipe.

        Args:
            anchor: Geographic extent and time window to search.
            geobox: Exact output grid for every raster.

        Returns:
            Raster name mapped to its selected lazy raster.

        Raises:
            ValueError: A raster declares no `stac` block, or a loaded raster
                misses its requirement.

        Examples:
            >>> spec.load_rasters(anchor, geobox=grid)["sentinel_2_l2a"].gs.variables
            ('B02', 'B03', 'B04', 'B08')
        """
        if missing := [n for n, r in self.rasters.items() if r.stac is None]:
            raise ValueError(f"Rasters declare no `stac` block: {missing}")
        rasters = {}
        for name, requirement in self.rasters.items():
            assert requirement.stac is not None
            try:
                rasters[name] = requirement.select_raster(
                    requirement.stac.load_raster(anchor, geobox=geobox)
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
        path: Location,
        *,
        geobox: GeoBox,
        path_format: str | Mapping[str, str] | None = None,
        id_format: str = ITEM_ID_FORMAT,
    ) -> GeoDataFrame:
        """Load every declared raster on an anchor, save it, and list it as Items.

        Training and inference ingest alike and get the same table. Pixels are
        saved as the source publishes them; `preprocess` runs when they are
        read. Each raster is saved as the collection its spec name says, so
        `row.gs.to_datatree().gs.rasters` hands `preprocess` the names it reads.

        Args:
            anchor: Geographic extent and time window to search.
            geobox: Exact output grid for every raster.
            path: Folder the COGs go under.
            path_format: Relative COG path, shared or keyed by raster. None
                writes `{collection}/{time}/{band}.tif`.
            id_format: `GeoAnchor.format` template each Item's id is filled
                from, as `Dataset.gs.to_catalog` takes it.

        Returns:
            Item table, one Item per raster and instant.

        Raises:
            ValueError: A raster declares no `stac` block, or misses its
                requirement.
            FileExistsError: A file the path format names exists.

        Examples:
            >>> items = spec.ingest(anchor, "data/s0", geobox=grid, path_format="{collection}/{band}.tif")
            >>> items.gs.sample()[0].gs.to_datatree().gs.groups
            ('sentinel_2_l2a',)
        """
        rasters = self.load_rasters(anchor, geobox=geobox)
        tree = xr.merge(
            [raster.gs.to_datatree(name) for name, raster in rasters.items()]
        )
        return tree.gs.to_catalog(path, path_format=path_format, id_format=id_format)

    def prepare(
        self,
        raw: GeoDataFrame,
        path: Location | None = None,
        *,
        target: Target | None = None,
        keep: Sequence[str] = (),
    ) -> GeoDataFrame:
        """Preprocess every sample of a raw catalog into a catalog of what the model reads.

        Each sample `raw.gs.sample()` groups by footprint is read on native
        grids unless `target` is given, then selected, validated and
        preprocessed. Raster dependencies of encoding, input binding, and decoding,
        plus collections named by `keep`, are saved under `path/<sample position>/`
        with this spec beside
        them. Windows are not cut here; sample the prepared catalog with
        `pair` and `window` to cut them.

        Args:
            raw: Item table of the declared rasters and the collections kept.
            path: New folder, as a path, URL, or UPath. None writes to a
                temporary folder, since prepared rasters serve this model only.
            target: Explicit alignment target; None keeps native grids.
            keep: Collections copied through unchanged, as labels for training.

        Returns:
            Item table of the prepared rasters, one Item per raster and instant.

        Raises:
            FileExistsError: `path` exists.
            TypeError: `keep` is given as text.
            KeyError: A sample lacks a declared raster or a kept collection.

        Examples:
            >>> prepared = spec.prepare(raw, "data/prepared", keep=["label"])
            >>> prepared.gs.sample()[0].gs.to_datatree().gs.groups
            ('image', 'label')
        """
        if isinstance(keep, str):
            raise TypeError(f"keep takes a list of collections, as keep=[{keep!r}]")
        if path is None:
            folder = storage.location(tempfile.mkdtemp(prefix="geosave-prepared-"))
        else:
            folder = storage.location(path)
            if folder.exists():
                raise FileExistsError(f"{folder} exists; prepare writes a new folder")
        kept = list(dict.fromkeys(keep))
        items = []
        # Collections the prepared rasters carry, as their samples' catalogs state them.
        carried: dict[str, pystac.Collection] = {}
        samples = raw.gs.sample()
        for position, rows in enumerate(samples):
            # A sample has no id of its own; its position names its folder.
            sample_id = str(position)
            tree = rows.gs.to_datatree()
            if target is not None:
                tree = tree.gs.align(target=target)
            rasters = {}
            for name in dict.fromkeys([*self.rasters, *kept]):
                if name in tree.children:
                    rasters[name] = tree[name].gs.to_dataset()
            if missing := [name for name in kept if name not in rasters]:
                raise KeyError(f"Sample {sample_id!r} lacks {missing}")
            results = self.preprocess(rasters)
            groups: dict[str, xr.Dataset] = {
                name: results[name] for name in self.input_rasters
            }
            for name in kept:
                groups[name] = rasters[name]
            # Samples share grids and dates, so the sample id keeps their Items apart.
            sample_tree = xr.merge(
                [group.gs.to_datatree(name) for name, group in groups.items()]
            )
            sample_catalog = sample_tree.gs.to_catalog(
                folder / sample_id,
                id_format=f"{sample_id}_{{start:%Y%m%dT%H%M%S}}",
                datetime=tree.gs.timespan,
            )
            items.extend(sample_catalog.gs.to_items())
            carried.update(sample_catalog.gs.collections)
        self.save(folder / self.filename)
        # The raw catalog's definitions come first; prepared rasters keep what they carry.
        definitions = {**carried, **cast("GeoCatalog", raw.gs).collections}
        collections = []
        for name in dict.fromkeys(item.collection_id for item in items):
            if name is not None and name in definitions:
                collections.append(definitions[name])
        return from_items(items, collections=collections)

    def encode(self, values: Mapping[str, Any], /) -> dict[str, Any]:
        """Encode one prepared sample, retaining its values for output decoding.

        Args:
            values: Results of preprocessing this sample. Encode after any native
                spatial/time selection that changes what the model will receive.

        Returns:
            Plain mapping of supplied and encoded values. Only explicitly declared
            calls compute pixels; intermediate state is not sent to the model.
        """
        return {**values, **self.encoding.run(values)}

    def model_inputs(self, values: Mapping[str, Any], /) -> dict[str, Any]:
        """Bind encoded values to model argument names without rerunning stages.

        Args:
            values: This sample's encoded values.

        Returns:
            Only the named model inputs. Call transform before model execution.
        """
        return {name: ref.resolve(values) for name, ref in self.inputs.items()}

    def transform(
        self, inputs: Mapping[str, Any], /, *, batched: bool = False
    ) -> dict[str, Any]:
        """Apply shared model-input transforms after optional training augmentation.

        Args:
            inputs: Named model arguments. Calls may rebind them, but intermediate
                results are not added to the returned model arguments.
            batched: Validate a leading DataLoader batch axis in tensor contracts.

        Returns:
            Transformed arguments, preserving undeclared extra arguments unchanged.
            Calls must support the stated batch layout; no axes are silently folded.

        Raises:
            ValueError: A required tensor has the wrong shape or dtype.
        """
        self._check_tensors(inputs, batched=batched)
        values = {**inputs, **self.transforms.run(inputs)}
        result = {name: values[name] for name in inputs}
        self._check_tensors(result, batched=batched)
        return result

    def _check_tensors(self, inputs: Mapping[str, Any], *, batched: bool) -> None:
        sizes: dict[str, int] = {}
        for name, requirement in self.tensors.items():
            requirement.validate_tensor(
                inputs[name], name=name, sizes=sizes, batched=batched
            )

    def decode(self, prediction: Any, values: Mapping[str, Any], /) -> dict[str, Any]:
        """Decode one model output using the same sample's intermediate state.

        Args:
            prediction: Unbatched model output, including native dictionaries/tuples.
            values: Values retained from this sample's encode call.

        Returns:
            Declared decoding results. The caller chooses what to merge.
        """
        return self.decoding.run({**values, "prediction": prediction})

    def postprocess(self, prediction: Any, /) -> dict[str, Any]:
        """Process a merged prediction, after all chip outputs have been accumulated.

        Args:
            prediction: Merged native output, or mapping of named merged outputs.

        Returns:
            Declared results, such as class IDs derived from merged logits.
        """
        return self.postprocessing.run({"prediction": prediction})

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
