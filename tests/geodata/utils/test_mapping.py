"""Merging several mappings key by key."""

from __future__ import annotations

import numpy as np

from geosave.geodata.utils.mapping import intersect_mappings, stack_mappings


def test_intersection_keeps_only_keys_every_mapping_states_alike() -> None:
    shared = intersect_mappings(
        [
            {"platform": "s2a", "gsd": 10, "instruments": ["msi"]},
            {"platform": "s2b", "gsd": 10, "instruments": ["msi"]},
            {"gsd": 10, "instruments": ["msi"], "extra": 1},
        ]
    )

    assert shared == {"gsd": 10, "instruments": ["msi"]}


def test_intersection_reads_nan_and_arrays_as_values() -> None:
    shared = intersect_mappings(
        [
            {
                "fill": float("nan"),
                "codes": np.array([1, 2]),
                "scale": np.float32(np.nan),
            },
            {
                "fill": float("nan"),
                "codes": np.array([1, 2]),
                "scale": np.float32(np.nan),
            },
        ]
    )

    assert set(shared) == {"fill", "codes", "scale"}


def test_stack_keeps_one_entry_per_mapping_in_first_seen_key_order() -> None:
    stacked = stack_mappings([{"platform": "s2a", "cloud": 10.0}, {"platform": "s2b"}])

    assert stacked == {"platform": ["s2a", "s2b"], "cloud": [10.0, None]}
