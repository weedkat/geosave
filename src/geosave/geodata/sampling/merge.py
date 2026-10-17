"""Reconstruct chip outputs on their recorded layout grids."""

from __future__ import annotations


# Work behind the tables' operations; call those instead.
__all__: list[str] = []

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import geopandas as gpd

import numpy as np
from tiler import Merger, Tiler

from geosave.geodata.conventions import SPATIAL_DIMENSIONS
from geosave.geodata.core.raster.create import raster

if TYPE_CHECKING:
    from affine import Affine
    from odc.geo.geobox import GeoBox

    from geosave.geodata import Dataset
    from geosave.geodata.core.vector.samples import Taper


def create_taper_weights(
    chip_shape: tuple[int, int], taper: Taper | None
) -> np.ndarray:
    """Return how much each chip pixel counts, as Tiler's window `taper` weighs it."""
    if taper is None:
        return np.ones(chip_shape)
    merger = Merger(Tiler(chip_shape, chip_shape), window=taper)
    return np.asarray(merger.window, dtype=np.float64)


def read_chip_output(
    chip_label: Hashable,
    outputs: Mapping[Hashable, np.ndarray],
    valid: Mapping[Hashable, np.ndarray] | None,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a chip's prediction with invalid pixels zeroed, and its 0/1 validity.

    Raises:
        ValueError: The chip has no output, or the output or mask has the wrong shape.
    """
    if chip_label not in outputs:
        raise ValueError(f"chip {chip_label!r} has no output")
    prediction = np.asarray(outputs[chip_label], dtype=np.float64)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"chip {chip_label!r} output has shape {prediction.shape}; "
            f"expected {expected_shape}"
        )
    is_valid = np.isfinite(prediction).all(axis=0)
    if valid is not None:
        if chip_label not in valid:
            raise ValueError(f"chip {chip_label!r} has no validity mask")
        mask = np.asarray(valid[chip_label], dtype=bool)
        if mask.shape != expected_shape[1:]:
            raise ValueError(
                f"chip {chip_label!r} validity has shape {mask.shape}; "
                f"expected {expected_shape[1:]}"
            )
        is_valid &= mask
    valid_prediction = np.asarray(np.where(is_valid, prediction, 0.0))
    return valid_prediction, np.asarray(is_valid, dtype=np.float64)


def find_chip_offset(region: GeoBox, chip_grid: GeoBox) -> tuple[int, int]:
    """Return the row and column of a chip's first pixel in the region's grid."""
    if region.crs != chip_grid.crs:
        raise ValueError("chip CRS differs from the parent grid")
    pixel_offset = cast("Affine", ~region.transform * chip_grid.transform)
    top, left = round(pixel_offset.f), round(pixel_offset.c)
    if not np.allclose(
        [
            pixel_offset.a,
            pixel_offset.b,
            pixel_offset.d,
            pixel_offset.e,
            pixel_offset.f,
            pixel_offset.c,
        ],
        [1, 0, 0, 1, top, left],
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError("chip grid is not on the parent's pixel lattice")
    return top, left


def add_chip(
    weighted_sums: np.ndarray,
    weight_sums: np.ndarray,
    prediction: np.ndarray,
    chip_weights: np.ndarray,
    chip_offset: tuple[int, int],
) -> None:
    """Add a chip's weighted prediction to the sums over the pixels it shares with the region."""
    region_height, region_width = weight_sums.shape
    chip_height, chip_width = chip_weights.shape
    top, left = chip_offset
    first_row, first_column = max(top, 0), max(left, 0)
    last_row = min(top + chip_height, region_height)
    last_column = min(left + chip_width, region_width)
    if first_row >= last_row or first_column >= last_column:
        return

    chip_rows = slice(first_row - top, last_row - top)
    chip_columns = slice(first_column - left, last_column - left)
    shared_weights = chip_weights[chip_rows, chip_columns]
    weighted_sums[:, first_row:last_row, first_column:last_column] += (
        prediction[:, chip_rows, chip_columns] * shared_weights
    )
    weight_sums[first_row:last_row, first_column:last_column] += shared_weights


def build_merged_raster(
    weighted_sums: np.ndarray,
    weight_sums: np.ndarray,
    bands: Sequence[str],
    region: GeoBox,
) -> Dataset:
    """Divide the sums into a mean per band; a pixel no chip weighs stays NaN."""
    means = np.full(weighted_sums.shape, np.nan)
    np.divide(weighted_sums, weight_sums, out=means, where=weight_sums > 0)
    variables = {}
    for band, band_means in zip(bands, means, strict=True):
        variables[band] = (SPATIAL_DIMENSIONS, band_means)
    return raster(variables, region)


def merge(
    samples: gpd.GeoDataFrame,
    chips: gpd.GeoDataFrame,
    outputs: Mapping[Hashable, np.ndarray],
    *,
    bands: Sequence[str],
    valid: Mapping[Hashable, np.ndarray] | None,
    taper: Taper | None,
) -> dict[Hashable, Dataset]:
    """Implement `GeoSamples.merge` without changing its public table contract."""
    if (
        not bands
        or len(set(bands)) != len(bands)
        or not all(name.strip() for name in bands)
    ):
        raise ValueError("bands must hold unique, nonempty names")
    if not samples.index.is_unique or not chips.index.is_unique:
        raise ValueError("sample and chip row labels must be unique")
    if "parent" not in chips or bool(chips["parent"].isna().any()):
        raise ValueError("chips need parent references; pass rows made by chip")

    parents = {}
    for label, row in samples.iterrows():
        key = (row.gs.key, row.gs.timespan)
        if key in parents:
            raise ValueError("samples must have unique spatial identities and spans")
        parents[key] = label

    chip_labels = {}
    for label, row in chips.iterrows():
        key = (row.at["parent"], row.gs.timespan)
        if key not in parents:
            raise ValueError(
                "chips were not cut from these samples; pass the table given to chip"
            )
        chip_labels.setdefault(parents[key], []).append(label)

    weights = {}
    merged = {}
    for label, row in samples.iterrows():
        labels = chip_labels.get(label, [])
        if not labels:
            continue
        region = chips.loc[labels[0]].gs.parent_grid
        if labels:
            for chip_label in labels:
                chip_region = chips.loc[chip_label].gs.parent_grid
                if chip_region is not None and chip_region != region:
                    raise ValueError(
                        "chips of one parent must share the same layout grid"
                    )
        weighted_sums = np.zeros((len(bands), *region.shape))
        weight_sums = np.zeros(tuple(region.shape))
        for chip_label in labels:
            grid = chips.loc[chip_label].gs.geobox
            offset = find_chip_offset(region, grid)
            shape = (int(grid.shape[0]), int(grid.shape[1]))
            if shape not in weights:
                weights[shape] = create_taper_weights(shape, taper)
            prediction, valid_weights = read_chip_output(
                chip_label, outputs, valid, (len(bands), *shape)
            )
            add_chip(
                weighted_sums,
                weight_sums,
                prediction,
                valid_weights * weights[shape],
                offset,
            )

        merged[label] = build_merged_raster(weighted_sums, weight_sums, bands, region)
    return merged
