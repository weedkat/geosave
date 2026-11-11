"""Frozen pre-release writer for historical benchmark comparisons only."""

from __future__ import annotations

import errno
import json
import os
from math import isqrt, prod
from operator import index as integer_index
from pathlib import Path
from shutil import disk_usage, rmtree
from typing import TYPE_CHECKING

import dask.config
import numpy as np
from tiler import Tiler

from geosave.geodata.io.chips import _close_map, _json, _manifest

if TYPE_CHECKING:
    from geosave.geodata.core.raster.chips import Chips


def _write(chips: Chips, path: Path, *, layout: str, workers: int) -> Path:
    workers = integer_index(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    if path.exists():
        raise FileExistsError(path)
    manifest = _manifest(chips, layout)
    # Serialize metadata before creating output or computing source pixels.
    encoded = json.dumps(manifest, default=_json)
    required = len(encoded.encode())
    for index, source in enumerate(manifest["sources"]):
        shape = source["shape"]
        pixels = prod(shape)
        if layout == "chips":
            tiler = chips._tilers[index]
            pixels = len(tiler) * prod(shape[:-2]) * prod(tiler.tile_shape[-2:])
        required += pixels * np.dtype(source["dtype"]).itemsize + 4096
    path.mkdir(parents=True)
    try:
        # Mapping a file larger than available disk can fail with SIGBUS while
        # writing. This preflight detects capacity shortages, not concurrent use.
        if disk_usage(path).free < required:
            raise OSError(errno.ENOSPC, "not enough disk space for chip store", path)
        with dask.config.set(scheduler="threads", num_workers=workers):
            for index, source in enumerate(manifest["sources"]):
                shape = tuple(source["shape"])
                dtype = np.dtype(source["dtype"])
                tiler = chips._tilers[index]
                if layout == "chips":
                    shape = (len(tiler), *shape[:-2], *map(int, tiler.tile_shape[-2:]))
                array = np.lib.format.open_memmap(
                    path / f"{index}.npy", mode="w+", shape=shape, dtype=dtype
                )
                try:
                    if layout == "sources":
                        side = max(
                            1,
                            isqrt(chips._max_block_bytes // chips._pixel_bytes[index]),
                        )
                        blocks = Tiler(shape[-2:], (side, side), mode="constant")
                        for block in range(len(blocks)):
                            low, high = blocks.get_tile_bbox(block)
                            top, left = map(int, low)
                            bottom = min(int(high[0]), shape[-2])
                            right = min(int(high[1]), shape[-1])
                            pixels = chips._read_window(
                                index, top, left, bottom - top, right - left
                            )
                            array[..., top:bottom, left:right] = pixels.reshape(
                                (*shape[:-2], bottom - top, right - left)
                            )
                            del pixels
                    else:
                        tile_bytes = prod(shape[1:]) * dtype.itemsize
                        count = max(1, chips._max_block_bytes // tile_bytes)
                        for start in range(0, len(tiler), count):
                            stop = min(start + count, len(tiler))
                            indices = range(
                                chips._offsets[index] + start,
                                chips._offsets[index] + stop,
                            )
                            pixels = chips.__getitems__(indices)
                            for tile, values in enumerate(pixels, start):
                                array[tile] = values
                                del values
                            del pixels
                    array.flush()
                finally:
                    _close_map(array)
                with (path / f"{index}.npy").open("rb") as file:
                    os.fsync(file.fileno())
        # A reader only accepts a store after every array is complete.
        with (path / "manifest.json").open("w") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        rmtree(path)
        raise
    return path
