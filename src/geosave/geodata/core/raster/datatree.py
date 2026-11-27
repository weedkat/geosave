"""General geospatial helpers for native xarray DataTrees."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack, cast, overload

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox


from . import rules as raster_rules
from .rules import datatree as tree_rules

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal
    from xarray.core.types import CombineAttrsOptions

    import pandas as pd
    import geopandas as gpd
    import torch

    from .utils.context import ContextReader, ContextTensors
    from odc.geo import SomeCRS, SomeResolution
    from odc.geo.geom import Geometry

    from geosave.geodata import Dataset, DataTree
    from dask.delayed import Delayed
    from geosave.geodata.io.raster.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.raster.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from upath import UPath
    from .transform.extent import Padding
    from .transform.warp import Resampling
    from datetime import datetime, timedelta

    from geosave.geodata.utils.datetime import DateRange, Freq

    from .transform.temporal import Interpolation, Reducer


def check_node_name(name: str) -> None:
    """Reject a name native DataTree would misread as a path.

    xarray accepts "", "." and ".." as child names and fails obscurely on "/".

    Raises:
        ValueError: The name is not one nonempty path component.
    """
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"Expected a child name, got {name!r}")


class GeoDataTree:
    """Organize georeferenced Dataset leaves on independently described grids."""

    def __init__(self, data: DataTree) -> None:
        """Bind the native tree without imposing a shared grid or timeline."""
        self._data = data

    @property
    def geobox(self) -> GeoBox:
        """Read the GeoBox every raster leaf shares.

        Leaves may sit on different grids; reading the tree's grid requires
        them to share one.

        Raises:
            RasterValidationError: The tree is invalid or its raster leaves
                sit on different GeoBoxes.

        Examples:
            >>> tree.gs.geobox.shape
            Shape2d(x=512, y=512)
        """
        self.validate(tree_rules.same_geobox)
        return cast("GeoBox", self._read_first_raster().gs.geobox)

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Read the spatial axes of the GeoBox every raster leaf shares, vertical then horizontal.

        Raises:
            RasterValidationError: The tree is invalid or its raster leaves
                sit on different GeoBoxes.

        Examples:
            >>> tree.gs.spatial_dims
            ('y', 'x')
        """
        self.validate(tree_rules.same_geobox)
        return self._read_first_raster().gs.spatial_dims

    @property
    def chunk_heights(self) -> tuple[int, ...] | None:
        """Read the Dask chunk heights every raster leaf shares, or None when pixels are not in Dask.

        Raises:
            RasterValidationError: The tree is invalid or its raster leaves
                are chunked differently.

        Examples:
            >>> tree.chunk(y=512).gs.chunk_heights
            (512, 512)
        """
        self.validate(tree_rules.same_chunks)
        return self._read_first_raster().gs.chunk_heights

    @property
    def chunk_widths(self) -> tuple[int, ...] | None:
        """Read the Dask chunk widths every raster leaf shares, or None when pixels are not in Dask.

        Raises:
            RasterValidationError: The tree is invalid or its raster leaves
                are chunked differently.

        Examples:
            >>> tree.chunk(x=512).gs.chunk_widths
            (512, 512)
        """
        self.validate(tree_rules.same_chunks)
        return self._read_first_raster().gs.chunk_widths

    @property
    def extent(self) -> Geometry | None:
        """Read the place the raster leaves declare together: the union of their extents.

        Returns:
            The union, in the first declaring leaf's CRS, or None when no leaf
            declares an extent.

        Examples:
            >>> tree.gs.crop(city).gs.extent == city
            True
        """
        union = None
        for node in self._data.leaves:
            if not node.data_vars:
                continue
            leaf_extent = node.gs.to_dataset().gs.extent
            if leaf_extent is None:
                continue
            if union is None:
                union = leaf_extent
            else:
                # A declared extent always carries its CRS.
                union = union.union(leaf_extent.to_crs(cast("SomeCRS", union.crs)))
        return union

    def _read_first_raster(self) -> Dataset:
        """Read the first raster leaf, which stands for all when a rule makes them alike."""
        first = next(node for node in self._data.leaves if node.data_vars)
        return first.gs.to_dataset()

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read the `time` labels every dated raster leaf shares.

        Returns:
            Labels in axis order, or None when no leaf has a `time` dimension.

        Raises:
            TypeError: A leaf's `time` dimension does not hold datetimes.
            ValueError: Dated leaves hold different labels.

        Examples:
            >>> tree.gs.times.strftime("%m-%d").tolist()
            ['06-03', '06-28']
        """
        times = None
        for node in self._data.leaves:
            node_times = node.gs.to_dataset().gs.times
            if node_times is None:
                continue
            if times is None:
                times = node_times
            elif not node_times.equals(times):
                raise ValueError(
                    f"node {node.path!r} has its own time labels; this operation "
                    "needs every dated node on one timeline"
                )
        return times

    @property
    def time_coordinate(self) -> xr.DataArray | None:
        """Read the `time` coordinate every dated raster leaf shares.

        Returns:
            The shared coordinate, or None when no leaf is dated. A scalar
            coordinate dates one observation, as for a Dataset.

        Raises:
            ValueError: Dated leaves hold different `time` coordinates.

        Examples:
            >>> scene.gs.time_coordinate.values
            numpy.datetime64('2025-06-03T00:00:00.000000000')
        """
        time = None
        for node in self._data.leaves:
            node_time = node.gs.to_dataset().gs.time_coordinate
            if node_time is None:
                continue
            if time is None:
                time = node_time
            elif not node_time.variable.equals(time.variable):
                raise ValueError(
                    f"node {node.path!r} has its own `time`; this operation needs "
                    "every dated node on one timeline"
                )
        return time

    @property
    def timespan(self) -> DateRange | None:
        """Return the earliest start and latest end over every dated raster leaf.

        Leaves may hold their own dates, such as two sensors observed on
        different days. Each leaf's span is its Dataset's `timespan`: the
        periods its `start_datetime` and `end_datetime` state, else its first
        and last label. Reads coordinates only, never pixels.

        Returns:
            The span, or None when no leaf is dated.

        Examples:
            >>> window.gs.timespan
            (datetime.datetime(2025, 6, 3, 0, 0), datetime.datetime(2025, 7, 23, 0, 0))
        """
        starts = []
        ends = []
        for node in self._data.leaves:
            span = node.gs.to_dataset().gs.timespan
            if span is not None:
                starts.append(span[0])
                ends.append(span[1])
        if not starts:
            return None
        return min(starts), max(ends)

    def to_dataset(self) -> Dataset:
        """Read this node with all inherited coordinates and Dataset attrs.

        Uses native ``to_dataset(inherit="all_coords")``, then merges attrs
        from root to leaf. Nearer ancestors override more distant ones, and
        this node's attrs take precedence. Variable attrs and encoding stay
        unchanged. Descendant data is not combined.

        Returns:
            A Dataset with a new attrs dictionary and shared, unevaluated
            pixel arrays. Parent updates appear on the next call; previously
            returned Datasets are not synchronized. Nested attr values are
            shared, following native shallow-copy behavior.

        Examples:
            >>> tree.attrs["license"] = "CC-BY-4.0"
            >>> raster = tree["image"].gs.to_dataset()
            >>> raster.attrs["license"]
            'CC-BY-4.0'
        """
        dataset = self._data.to_dataset(inherit="all_coords")

        inherited_attrs = {}
        for parent in reversed(self._data.parents):
            inherited_attrs.update(parent.attrs)

        inherited_attrs.update(dataset.attrs)
        dataset.attrs = inherited_attrs

        return cast("Dataset", dataset)

    def group(self, groups: Mapping[str, list[str]]) -> DataTree:
        """Partition this raster leaf's variables into named child leaves.

        Uses the same complete, exclusive partition as Dataset `to_tree(groups=)`.
        Variable dimensions and leaf grids remain unchanged. Shared Dataset attrs
        stay on the new parent, which retains this node's name.

        Args:
            groups: Ordered child names mapped to variable-name lists.

        Returns:
            New subtree with shared lazy arrays and unchanged source data.
            Assign it back explicitly to replace the original leaf.

        Raises:
            ValueError: The node is not a raster leaf, group names are invalid,
                or variable assignments are empty, unknown, repeated or incomplete.

        Examples:
            >>> grouped = tree["s2"].gs.group({"image": ["B04"], "scl": ["SCL"]})
            >>> restored = grouped.gs.merge()
        """
        self.validate(tree_rules.one_raster)
        raster = self.to_dataset()

        # Every variable goes to exactly one group, in the requested order.
        assigned = set()
        children = {}
        for name, variables in groups.items():
            check_node_name(name)
            if not isinstance(variables, list) or not variables:
                raise ValueError(f"Group {name!r} needs a nonempty variable-name list")
            for variable in variables:
                if variable not in raster.data_vars:
                    raise ValueError(f"Unknown variable {variable!r} in group {name!r}")
                if variable in assigned:
                    raise ValueError(
                        f"Variable {variable!r} is assigned more than once"
                    )
                assigned.add(variable)

            child = raster[variables].assign_coords(raster.coords)
            child.attrs = {}
            children[name] = xr.DataTree(child)

        missing = set(raster.data_vars) - assigned
        if missing:
            raise ValueError(
                f"Variables missing from groups: {sorted(missing, key=str)}"
            )

        return self._build_group(raster, children)

    def split(self, dim: str, *, names: list[str] | None = None) -> DataTree:
        """Split a raster leaf by dimension position, keeping static data once.

        The new parent retains this node's name and effective attrs. Each slice
        keeps its grid and original scalar coordinate under a subgroup named
        after `dim`. Variables without `dim` stay in a sibling named `static`.

        Args:
            dim: Non-spatial dimension to split by position.
            names: One unique child name per position. None uses coordinate
                labels as strings; repeated labels require explicit names.

        Returns:
            New subtree with shared lazy arrays and unchanged source data.
            Concat the dimension subgroup to reconstruct the sliced variables.

        Raises:
            RasterValidationError: The node is not a raster leaf, the dimension
                is spatial, has no one-dimensional coordinate, or no variable
                uses it.
            ValueError: Slice names are invalid, repeated or not one per position.

        Examples:
            >>> split = tree["s2"].gs.split("time", names=["first", "second"])
            >>> series = split["time"].gs.concat("time")
        """
        self.validate(tree_rules.splittable(dim))
        raster = self.to_dataset()

        if names is None:
            names = [str(label) for label in raster[dim].values]

        varying = []
        for key, variable in raster.data_vars.items():
            if dim in variable.dims:
                varying.append(key)

        # One leaf per position of the variables that vary along `dim`.
        observations = raster.drop_vars(set(raster.data_vars) - set(varying))
        observations.attrs = {}
        slices = {}
        for position, name in enumerate(names):
            check_node_name(name)
            slices[name] = xr.DataTree(observations.isel({dim: position}))
        if len(names) != raster.sizes[dim] or len(slices) != len(names):
            raise ValueError(
                "names must give one unique child name per dimension position; "
                "provide explicit names for repeated labels"
            )

        # Variables without `dim` stay once, beside the slices.
        children = {}
        static = raster.drop_vars(varying).drop_dims(dim)
        if static.data_vars:
            static.attrs = {}
            children["static"] = xr.DataTree(static)
        slice_metadata = xr.Dataset()
        slice_metadata.encoding = dict(raster.encoding)
        children[dim] = xr.DataTree(slice_metadata, children=slices)

        return self._build_group(raster, children)

    def concat(
        self, dim: str, *, combine_attrs: CombineAttrsOptions = "no_conflicts"
    ) -> DataTree:
        """Concatenate corresponding variables from every immediate raster child.

        Child order determines axis order, including repeated labels. Each
        child must carry `dim` as a scalar coordinate or existing dimension.
        Variable sets, grids, other indexes and pixel representations must
        agree. No dtype promotion, decoding or missing-variable insertion occurs.

        Args:
            dim: Coordinate to concatenate along; node names do not supply it.
            combine_attrs: Native xarray policy for the children's Dataset
                attrs. Variables and coordinates keep the first child's attrs,
                whose pixel meaning the rule has checked across children.

        Returns:
            New raster leaf named after this selected group. Pixels stay lazy;
            coordinate values may be compared by native xarray concatenation.

        Raises:
            RasterValidationError: Children are not compatible raster leaves, labels are
                absent, or corresponding variable representations differ.
            ValueError: Native xarray cannot combine Dataset attrs or coordinates.
        """
        self.validate(tree_rules.compatible_concat(dim))
        datasets: list[xr.Dataset] = []
        for child in self._data.children.values():
            datasets.append(child.gs.to_dataset())

        # "override" keeps the first child's variable and coordinate attrs;
        # the caller's policy then combines only the Dataset attrs.
        result = xr.concat(
            datasets,
            dim=dim,
            data_vars="all",
            coords="different",
            join="exact",
            compat="equals",
            combine_attrs="override",
        )
        dataset_attrs = []
        for dataset in datasets:
            dataset_attrs.append(xr.Dataset(attrs=dataset.attrs))
        result.attrs = xr.merge(dataset_attrs, combine_attrs=combine_attrs).attrs
        result.encoding = dict(self._data.encoding)
        return cast("DataTree", xr.DataTree(result, name=self._data.name))

    def merge(self, *, combine_attrs: CombineAttrsOptions = "no_conflicts") -> DataTree:
        """Merge all immediate raster children into one Dataset leaf.

        Spatial grids and shared dimension indexes must agree exactly. Each
        variable name must occur once. Different variables keep their own units,
        dtypes and encoding. Neither pixels nor the source tree are changed.

        Args:
            combine_attrs: Native xarray policy for effective Dataset attrs.
                Defaults to preserving all attrs and rejecting conflicts.

        Returns:
            New leaf with this node's name, combined data and no children.

        Raises:
            RasterValidationError: A child is a group, grids differ or variable
                names repeat. Select and merge nested groups first.
            ValueError: Native xarray cannot combine coordinates or attrs.

        Examples:
            >>> merged = tree["s2"].gs.merge()
            >>> dataset = merged.gs.to_dataset()
        """
        self.validate(tree_rules.compatible_merge)
        datasets: list[xr.Dataset] = []
        for child in self._data.children.values():
            datasets.append(child.gs.to_dataset())

        dataset = xr.merge(
            datasets, join="exact", compat="equals", combine_attrs=combine_attrs
        )
        for source in datasets:
            for key, variable in source.data_vars.items():
                dataset[key].attrs = dict(variable.attrs)
        dataset.encoding = dict(self._data.encoding)

        return cast("DataTree", xr.DataTree(dataset, name=self._data.name))

    def ungroup(self, name: str) -> DataTree:
        """Remove one direct child group, promoting its children in place.

        The removed group's attrs and inherited coordinates stay available on
        the promoted subtrees. Other siblings keep their own metadata. Shared
        parent attrs retain their parent owner.

        Args:
            name: Direct child group to remove, without a path or pattern.

        Returns:
            New containing subtree with shared pixel arrays and the same order,
            replacing the selected group with its children.

        Raises:
            KeyError: The named direct child does not exist.
            ValueError: The child is not a group or a promoted name collides.
            RasterValidationError: The tree structure is invalid.

        Examples:
            >>> result = tree.gs.ungroup("s2")
        """
        group = self._data.children[name]
        if not group.children or group.data_vars:
            raise ValueError(f"{name!r} must be a group without raster variables")

        self.validate()
        occupied = (set(self._data.children) - {name}) | set(self._data.variables)
        collisions = occupied.intersection(group.children)
        if collisions:
            raise ValueError(
                f"Promoted names already exist: {sorted(collisions, key=str)}"
            )

        result = self._data.copy(deep=False)
        result.dataset = self.to_dataset()
        children = {}
        for key, child in result.children.items():
            if key != name:
                children[key] = child
                continue

            for promoted_name, descendant in child.children.items():
                dataset = descendant.to_dataset(inherit="all_coords")
                dataset.attrs = {**child.attrs, **descendant.attrs}
                promoted = descendant.copy(deep=False)
                promoted.dataset = dataset
                children[promoted_name] = promoted

        result.children = children
        return cast("DataTree", result)

    def _build_group(
        self, raster: xr.Dataset, children: Mapping[str, xr.DataTree]
    ) -> DataTree:
        """Return the group replacing this leaf: its name, Dataset attrs and encoding over `children`."""
        metadata = xr.Dataset(attrs=raster.attrs)
        metadata.encoding = dict(raster.encoding)
        return cast(
            "DataTree", xr.DataTree(metadata, name=self._data.name, children=children)
        )

    def validate(self, *rules: raster_rules.Rule[DataTree]) -> None:
        """Apply `raster_leaves`, then additional rules, without reading pixels.

        Rasters must be leaves, each with its own grid ODC recognizes and a
        CRS stored locally on that leaf. Organizational parents may carry attrs
        and non-spatial coordinates; grids are not inherited. A root raster is
        valid when it has no children. A tree without rasters fails.

        Args:
            *rules: Tree rules the operation needs, such as `same_grid` or
                `same_geobox`.

        Raises:
            RasterValidationError: A raster or a rule is broken. The error
                names the node and how to fix it.

        Examples:
            >>> tree.gs.validate(same_geobox)
        """
        for rule in (tree_rules.raster_leaves, *rules):
            rule(self._data)

    def crop(
        self, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
    ) -> DataTree:
        """Cut every raster node down to the extent of a region.

        Args:
            region: Geometry, or geometries, to cut against, reprojected onto
                each node's CRS where they sit in another.
            mask: Also make nodata the pixels outside the geometries.

        Returns:
            New tree with cropped leaf grids and unchanged grouping metadata.

        Examples:
            >>> tree.gs.crop(field_boundaries)
        """
        self.validate()
        tree = self._data.copy(deep=False)
        tree.dataset = self.to_dataset()

        def cut(raster: xr.Dataset) -> xr.Dataset:
            if not raster.data_vars:
                return raster
            return raster.gs.crop(region, mask=mask)

        return cast("DataTree", tree.map_over_datasets(cut))

    def crop_to_grid(
        self,
        geobox: GeoBox,
        *,
        mode: Padding = "constant",
        constant_value: int | float = 0,
    ) -> DataTree:
        """Take the pixels of an aligned grid from every raster node, padding past the edges.

        Args:
            geobox: Grid sharing the rasters' CRS and pixel size, offset by
                whole pixels. It may extend past the raster edges.
            mode: How pixels outside a raster are filled, as in `numpy.pad`.
            constant_value: Fill for `"constant"`, keeping each dtype.

        Returns:
            New tree with every raster node on exactly `geobox`. Declared extents
            are clipped at their existing owner, including shared group attrs.
            Group coordinates are preserved. Dask-backed pixels stay lazy through padding.

        Raises:
            ValueError: A raster has no regular grid, or `geobox` is not
                pixel-aligned with it or does not overlap it.

        Examples:
            >>> chip = tree.gs.crop_to_grid(reference.gs.geobox[0:256, 0:256])
        """

        self.validate()
        tree = self._data.copy(deep=False)
        tree.dataset = self.to_dataset()

        def cut(raster: xr.Dataset) -> xr.Dataset:
            if not raster.data_vars:
                return raster.gs.clip_extent(geobox.extent)
            return raster.gs.crop_to_grid(
                geobox, mode=mode, constant_value=constant_value
            )

        return cast("DataTree", tree.map_over_datasets(cut))

    @overload
    def to_numpy(self, *, context: None = None) -> dict[str, np.ndarray]: ...

    @overload
    def to_numpy(
        self, *, context: Mapping[str, ContextReader[DataTree] | str]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...

    def to_numpy(
        self, *, context: Mapping[str, ContextReader[DataTree] | str] | None = None
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read each raster node's pixels, keyed by node path, and the requested context.

        Each raster node's pixels are stacked as `Dataset.gs.to_numpy` stacks
        them and keyed by the node's path, the model parameter they feed.
        Context is read before any pixel; all nodes are then read in one
        computation.

        Args:
            context: Model parameter names mapped to the readers of their
                arrays. A reader is a function of this tree, or its name in
                `CONTEXT_READERS`, such as `"transform"`, `"crs"` or
                `"centroid"`. None reads no context.

        Returns:
            Pixel arrays keyed by node path; with `context`, the pixel arrays
            and the context arrays keyed as `context`.

        Raises:
            RasterValidationError: The tree is invalid or grid context meets
                leaves on different grids.
            KeyError: A reader name is not in `CONTEXT_READERS`.
            ValueError: A raster cannot be stacked, or a reader refuses the tree.

        Examples:
            >>> tree.gs.to_numpy()["image"].shape
            (1, 4, 256, 256)
            >>> pixels, context = tree.gs.to_numpy(context={"centroid": "centroid"})
        """
        from .utils.context import read_context

        self.validate()
        values = None if context is None else read_context(self._data, context)

        tree = self._data.compute()
        pixels = {}
        for node in tree.subtree:
            if node.data_vars:
                path = node.relative_to(tree)
                pixels[path] = tree[path].gs.to_dataset().gs.to_numpy()
        if values is None:
            return pixels
        return pixels, values

    @overload
    def to_tensor(self, *, context: None = None) -> dict[str, torch.Tensor]: ...

    @overload
    def to_tensor(
        self, *, context: Mapping[str, ContextReader[DataTree] | str]
    ) -> tuple[dict[str, torch.Tensor], ContextTensors]: ...

    def to_tensor(
        self, *, context: Mapping[str, ContextReader[DataTree] | str] | None = None
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ContextTensors]:
        """Read each raster node's pixels and the requested context as `to_numpy` does, as tensors.

        Args:
            context: Model parameter names mapped to readers, as for `to_numpy`.

        Returns:
            Pixel tensors keyed by node path; with `context`, the pixel tensors
            and the context tensors keyed as `context`.

        Raises:
            RasterValidationError: As for `to_numpy`.
            KeyError: As for `to_numpy`.
            ValueError: As for `to_numpy`.
            TypeError: An array's dtype has no tensor equivalent.

        Examples:
            >>> pixels, context = tree.gs.to_tensor(context={"centroid": "centroid"})
            >>> logits = model(**pixels, **context)
        """
        from .utils.tensor import as_tensors

        if context is None:
            return as_tensors(self.to_numpy())

        pixels, values = self.to_numpy(context=context)
        return as_tensors(pixels), as_tensors(values)

    def mask_and_scale(self) -> DataTree:
        """Decode every raster node's CF values with `Dataset.gs.mask_and_scale`.

        Each node decodes its own packing and fill. Nodes without either, such
        as class IDs, keep their values and dtype. A decoded variable's dtype
        follows its packing, as `xarray.decode_cf` sets it.

        Returns:
            Lazy tree with unchanged grouping metadata and decoded raster leaves.

        Examples:
            >>> decoded = tree.gs.mask_and_scale()
            >>> monthly = decoded.gs.composite("MS", "median")
        """
        self.validate()

        def decode(raster: xr.Dataset) -> xr.Dataset:
            if not raster.data_vars:
                return raster
            return raster.gs.mask_and_scale()

        return cast("DataTree", self._data.map_over_datasets(decode))

    def composite(
        self,
        freq: Freq,
        method: Reducer,
        *,
        dim: str = "time",
        interpolation: Interpolation | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | datetime = "start_day",
        offset: str | timedelta | None = None,
    ) -> DataTree:
        """Composite every raster node with `Dataset.gs.composite`.

        Each dated node is reduced on its own timeline into the same calendar
        bins, so two sensors observed on different dates meet in one bin.
        Variables without `dim`, such as a static elevation or label, pass
        through unchanged.

        Args:
            freq: Pandas offset alias, such as "MS".
            method: One native reducer for every dated node; class IDs
                permit only first or last.
            dim: Datetime dimension to resample, default "time".
            interpolation: Fill interior pixel gaps after reduction, as for
                `Dataset.gs.composite`. None keeps gaps.
            closed: Inclusive bin edge, following native resample defaults.
            label: Bin label edge, following native resample defaults.
            origin: Native resample origin.
            offset: Shift applied to the origin.

        Returns:
            Lazy tree with unchanged grouping metadata and each dated node on
            its bins, stated by `start_datetime`/`end_datetime`.

        Raises:
            ValueError: A dated node holds packed values that need
                `mask_and_scale()` first, class IDs would be reduced
                numerically, or a node inherits `dim` from its parent.

        Examples:
            >>> monthly = tree.gs.composite("MS", "median")
        """
        self.validate()
        for node in self._data.subtree:
            inherited = node.to_dataset(inherit=True).xindexes
            own = node.to_dataset(inherit=False).xindexes
            if node.data_vars and dim in inherited and dim not in own:
                raise ValueError(
                    f"node {node.path!r} inherits `{dim}` from its parent; "
                    f"give each dated node its own `{dim}` before compositing"
                )

        tree = self._data.copy(deep=False)
        tree.dataset = self.to_dataset()

        def reduce(raster: xr.Dataset) -> xr.Dataset:
            if not raster.data_vars:
                return raster
            return raster.gs.composite(
                freq,
                method,
                dim=dim,
                interpolation=interpolation,
                closed=closed,
                label=label,
                origin=origin,
                offset=offset,
            )

        return cast("DataTree", tree.map_over_datasets(reduce))

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resolution: SomeResolution | None = None,
        resampling: Resampling = "nearest",
    ) -> DataTree:
        """Reproject every raster node with `Dataset.gs.reproject`.

        A GeoBox puts every node on that one grid; a CRS sizes each node's own
        grid, as `Dataset.gs.reproject` does.

        Args:
            how: GeoBox or CRS, as for `Dataset.gs.reproject`.
            resolution: Output pixel size for a CRS, as for `Dataset.gs.reproject`.
            resampling: One ODC resampling method for every raster node.

        Returns:
            Tree with unchanged grouping metadata and reprojected leaf grids.

        Examples:
            >>> aligned = tree.gs.reproject(labels.gs.geobox)
        """

        self.validate()
        tree = self._data.copy(deep=False)
        tree.dataset = self.to_dataset()

        def move(raster: xr.Dataset) -> xr.Dataset:
            if not raster.data_vars:
                return raster
            return raster.gs.reproject(
                how, resolution=resolution, resampling=resampling
            )

        return cast("DataTree", tree.map_over_datasets(move))

    def to_zarr(
        self,
        path: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one Zarr store with a group per node.

        Args:
            path: The store, ending in .zarr.
            compute: False returns one native `dask.Delayed`; computing it
                writes the store and returns its path.
            overwrite: Replace an existing store.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray Zarr write options; `encoding` is keyed by variable name and reaches every node holding it.

        Returns:
            Path of the store written, or one delayed result.

        Raises:
            ValueError: The destination does not end in .zarr.
            KeyError: An encoding name is no variable of the tree.
            FileExistsError: The store exists and overwrite is false.

        Examples:
            >>> open_raster_tree(tree.gs.to_zarr("scene.zarr")).children.keys()
        """
        from geosave.geodata.io.raster import zarr

        return zarr.write_zarr_tree(
            self._data,
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )

    def to_netcdf(
        self,
        path: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one netCDF file with a group per node.

        Args:
            path: The file, ending in .nc, .nc4 or .cdf.
            compute: False returns one native `dask.Delayed`; computing it
                writes the file and returns its path.
            overwrite: Replace an existing file.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray netCDF write options; `encoding` is keyed by variable name and reaches every node holding it.

        Returns:
            Path of the file written, or one delayed result.

        Raises:
            ValueError: The destination does not end in a netCDF suffix.
            KeyError: An encoding name is no variable of the tree.
            FileExistsError: The file exists and overwrite is false.

        Examples:
            >>> open_raster_tree(tree.gs.to_netcdf("scene.nc")).children.keys()
        """
        from geosave.geodata.io.raster import netcdf

        return netcdf.write_netcdf_tree(
            self._data,
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )
