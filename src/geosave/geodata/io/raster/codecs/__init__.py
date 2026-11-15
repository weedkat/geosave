"""Codecs: one in-memory convention translated to a stored one, and back.

A format lists the codecs its files need. Writing applies each `encode` in
order; reading applies each `decode` in reverse, so the pair cancels out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import xarray as xr


def keep(raster: xr.Dataset) -> xr.Dataset:
    """Return the raster as it is, for a convention that needs no decoding."""
    return raster


@dataclass(frozen=True)
class Codec:
    """One convention as written to a file and as read back from it.

    Attributes:
        encode: Spell the convention the way the format stores it.
        decode: Restore what `encode` changed.

    Examples:
        >>> TIME_BOUNDS = Codec(encode_time_bounds, decode_time_bounds)
    """

    encode: Callable[[xr.Dataset], xr.Dataset]
    decode: Callable[[xr.Dataset], xr.Dataset] = keep


def encode_dataset(raster: xr.Dataset, codecs: Sequence[Codec]) -> xr.Dataset:
    """Apply each codec's `encode`, first to last.

    Args:
        raster: Dataset about to be written.
        codecs: The format's codecs.

    Returns:
        Dataset spelled the way the format stores it.
    """
    for codec in codecs:
        raster = codec.encode(raster)
    return raster


def decode_dataset(raster: xr.Dataset, codecs: Sequence[Codec]) -> xr.Dataset:
    """Apply each codec's `decode`, last to first.

    Args:
        raster: Dataset as a reader opened it.
        codecs: The format's codecs, in the order they were encoded.

    Returns:
        Dataset as it was before `encode_dataset`.
    """
    for codec in reversed(codecs):
        raster = codec.decode(raster)
    return raster


def encode_tree(tree: xr.DataTree, codecs: Sequence[Codec]) -> xr.DataTree:
    """Encode every node's own Dataset.

    Args:
        tree: Tree about to be written.
        codecs: The format's codecs.

    Returns:
        Tree of the same groups, each encoded on its own.
    """
    nodes = {}
    for node in tree.subtree:
        nodes[node.path] = encode_dataset(node.to_dataset(inherit=False), codecs)
    return xr.DataTree.from_dict(nodes)


def decode_tree(tree: xr.DataTree, codecs: Sequence[Codec]) -> xr.DataTree:
    """Decode every node's own Dataset.

    Args:
        tree: Tree as a reader opened it.
        codecs: The format's codecs, in the order they were encoded.

    Returns:
        Tree of the same groups, each decoded on its own.
    """
    nodes = {}
    for node in tree.subtree:
        nodes[node.path] = decode_dataset(node.to_dataset(inherit=False), codecs)
    return xr.DataTree.from_dict(nodes)
