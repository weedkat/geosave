"""The `gs` accessor of a sample table: rows that each read as one DataTree.

`GeoCatalog.sample` makes a sample table from an item table. `frame` cuts each
sample into requested relative time bins, `chip` cuts a chosen grid into tiles, in
either order, and `merge` puts chip outputs back. Frames and chips retain the
sample schema; chips also record their parent for reconstruction.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from geosave.geodata.transform.temporal import Reducer

from .table import CollectionTable

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    from upath import UPath

    from geosave.geodata import Dataset, GeoDataFrame
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.transform.extent import Padding
    from geosave.geodata.transform.warp import Resampling


class Window(TypedDict):
    """One collection's ordered temporal bins and optional preparation."""

    slots: list[str]
    reduce: NotRequired[Reducer]
    fill: NotRequired[Literal["interpolate"]]


# How overlapping chips are weighed, as the window `tiler` names.
type Taper = Literal[
    "boxcar",
    "hamming",
    "triang",
    "bartlett",
    "hann",
    "barthann",
    "bohman",
    "parzen",
    "blackman",
    "nuttall",
    "blackmanharris",
    "overlap-tile",
]


class GeoSamples(CollectionTable):
    """Operate on spatial samples, temporal frames and pixel chips.

    Required fields are id, area (WGS84 GeoJSON), collections, serialized
    Items, and start/end dates. The area remains stable when the displayed
    geometry is renamed or reprojected. Samples have no pixel grid.

    Chips additionally carry proj:code, proj:transform, proj:shape, resampling,
    padding, parent identity and parent_grid. Caller columns and Collection metadata
    survive framing, chipping and GeoParquet persistence.

    Examples:
        >>> samples = items.gs.sample()
        >>> frames = samples.gs.frame(windows={"image": {"slots": ["-3M", "-2M", "-1M"]}})
        >>> chips = frames.gs.chip(224, target=grid, overlap=32)
    """

    # Data fields identify samples independently of the active geometry name.
    FIELDS = (
        "id",
        "area",
        "collections",
        "items",
        "start_datetime",
        "end_datetime",
    )
    COLUMNS = (*FIELDS, "geometry", "parent")

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[Hashable, Any] | pd.Series],
        *,
        crs: Any = "EPSG:4326",
        geometry: Hashable = "geometry",
        attrs: Mapping[Hashable, Any] | None = None,
    ) -> GeoDataFrame[GeoSamples]:
        """Build a sample table from row records.

        Args:
            rows: One mapping of column to value per row, in order. Columns
                beyond `COLUMNS`, such as a caller's `split`, are kept after them.
            crs: CRS of the geometry values.
            geometry: Active geometry column name.
            attrs: Table metadata, copied onto the result.

        Returns:
            Sample table with a fresh RangeIndex.
        """
        row_records = [dict(row) for row in rows]
        column_names: list[Hashable] = [*cls.FIELDS, geometry, "parent"]
        for row_record in row_records:
            for column_name in row_record:
                if column_name not in column_names:
                    column_names.append(column_name)
        table = gpd.GeoDataFrame(
            row_records,
            columns=pd.Index(column_names),
            geometry=geometry,
            crs=crs,
        )
        table.attrs = dict(attrs or {})  # pyright: ignore[reportAttributeAccessIssue]
        return cast("GeoDataFrame[GeoSamples]", table)

    def frame(
        self,
        *,
        windows: Mapping[str, Window],
        reference: str | None = None,
        stride: int | None = None,
    ) -> GeoDataFrame[GeoSamples]:
        """Build complete model inputs from relative temporal bins.

        Args:
            windows: Per-collection slots and optional reduce/fill settings.
                '-1W' covers the seven days before the reference, start included
                and end excluded. M and Y mean relative calendar months/years.
                Composite coverage must fit wholly inside a bin. Multiple
                acquisitions require a reducer; first/last select whole scenes.
                fill='interpolate' fills interior missing bins only.
            reference: Collection supplying reference timestamps and observations.
                None uses the union of requested collections' timestamps.
            stride: Steps through newest-first reference timestamps, before
                completeness checks. None uses the largest slot count.

        Returns:
            Sample rows with persisted bin selections and lazy preparation.
            Incomplete frames are omitted with a warning; pixels are not read.

        Raises:
            ValueError: Settings or collection names are invalid, slots overlap,
                or multiple acquisitions have no explicit reduction policy.

        Examples:
            >>> frames = samples.gs.frame(
            ...     windows={"image": {"slots": ["-2W", "-1W"], "reduce": "first"}},
            ...     reference="label", stride=1,
            ... )
        """
        from geosave.geodata.sampling.frame import frame

        return frame(self._data, windows=windows, reference=reference, stride=stride)

    def chip(
        self,
        size: int | tuple[int, int],
        *,
        resolution: float | None = None,
        target: GeoBox | None = None,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        overlap: int | float = 0,
        pad_mode: Padding = "reflect",
    ) -> GeoDataFrame[GeoSamples]:
        """Plan aligned pixel chips at a chosen resolution without reading files.

        Chips cover the row's read region on the chosen pixel lattice.
        Chipping a chip cuts its read part. Pixels past the region are filled as
        `pad_mode` says when read. No file is opened::

            region 50 x 70, chip 32, overlap 8
            +--------+--------+--------+
            |pad     |pad     |pad     |   pad_widths [[12, 0], [12, 0]]
            |  +-----+--------+-----+  |   for the first chip
            |  |     region         |  |
            +--+--------------------+--+

        Args:
            size: Chip side in pixels, or height and width.
            resolution: Positive pixel size in CRS units. Each row uses its
                sources' shared CRS, with pixel edges snapped to multiples of
                this resolution. Different rows can use different CRSs. Sources
                within a row must state one shared CRS; otherwise supply target.
            target: Grid supplying CRS, resolution and pixel alignment. Its
                lattice is extended over each row's read region; its extent
                does not limit coverage. Cannot be combined with resolution.
                With neither option, an existing chip's grid is reused.
            resampling: Kernel or collection/variable mapping used when chips
                are read. Nearest preserves categorical label values by default.
            overlap: Pixels neighbouring chips share (int), or a fraction of
                each chip dimension in [0.0, 1.0) (float). Fractional overlap
                is rounded up to whole pixels by Tiler.
            pad_mode: How a chip's pixels past the read region are filled.

        Returns:
            Sample table, one row per chip in Tiler's order: its own grid,
            footprint, `pad_widths` and `pad_mode`.

        Examples:
            >>> chips = samples.gs.chip(32, resolution=10, overlap=8)
            >>> chips[["proj:shape", "pad_widths"]].iloc[0].tolist()
            [[32, 32], [[12, 0], [12, 0]]]
        """
        from geosave.geodata.sampling.chip import chip

        return chip(
            self._data,
            size,
            target=target,
            resolution=resolution,
            resampling=resampling,
            overlap=overlap,
            pad_mode=pad_mode,
        )

    def merge(
        self,
        chips: gpd.GeoDataFrame,
        outputs: Mapping[Hashable, np.ndarray],
        *,
        bands: Sequence[str],
        valid: Mapping[Hashable, np.ndarray] | None = None,
        taper: Taper | None = None,
    ) -> dict[Hashable, Dataset]:
        """Merge chip outputs onto their recorded layout grid for each sample.

        A chip records its parent's spatial key and must match its time span.
        Its grid must share the recorded layout's CRS and pixel lattice.
        Each pixel is the weighted mean of the valid chip outputs covering it,
        weighed by `taper`; a pixel
        none covers validly is NaN. Overlaps average as logits do, so take
        classes after merging.

        Args:
            chips: Chips cut from these samples, or a subset of them.
            outputs: Each chip's row label mapped to its `(band, y, x)` output.
            bands: Name of each output band, in order.
            valid: Each chip's row label mapped to its `(y, x)` pixels that
                count. None counts every finite pixel.
            taper: Tiler window weighing each chip's pixels, such as `"hann"`.
                None weighs every pixel alike. A tapered edge counts for
                almost nothing, so taper only chips cut with overlap.

        Returns:
            {
                <sample row label>: float64 Dataset on the chip layout grid, one
                    `(y, x)` variable per band,
            }

        Raises:
            ValueError: Bands repeat or are empty, row labels or parent identities
                are ambiguous, a chip matches no sample or pixel lattice, or a
                chip's output or validity is missing or misshapen.

        Examples:
            >>> merged = samples.gs.merge(chips, outputs, bands=["water", "trees"])
            >>> merged[0].to_dataarray("band").argmax("band")
            <xarray.DataArray (y: 512, x: 512)> ...
        """
        from geosave.geodata.sampling.merge import merge

        return merge(self._data, chips, outputs, bands=bands, valid=valid, taper=taper)

    def to_geoparquet(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Any,
    ) -> UPath:
        """Write this sample table as GeoParquet, its Collections in the file's metadata.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **options: Not accepted; the sample encoding is fixed.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath
            carrying its storage options.

        Raises:
            FileExistsError: The file exists and `overwrite` is false.
            ValueError: The suffix is wrong, or options are passed.

        Examples:
            >>> chips.gs.to_geoparquet("data/chips.parquet")
            PosixUPath('data/chips.parquet')
            >>> gs.open_vector("data/chips.parquet").gs.collections.keys()
            dict_keys(['label', 's2'])
        """
        if options:
            raise ValueError(
                f"a sample table fixes its own encoding; {sorted(options)} do not apply"
            )
        from geosave.geodata.io.vector import sample_geoparquet

        return sample_geoparquet.write(
            self._data, path, overwrite=overwrite, storage_options=storage_options
        )
