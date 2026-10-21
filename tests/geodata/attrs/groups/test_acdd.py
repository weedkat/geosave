from __future__ import annotations

import xarray as xr

from geosave.geodata import attrs
from geosave.geodata.attrs import ACDD


def test_acdd_reads_only_its_own_keys() -> None:
    assert ACDD.from_attrs({"title": "Forest", "units": "1"}) == ACDD(title="Forest")


def test_attrs_stating_no_acdd_key_read_as_none() -> None:
    assert ACDD.from_attrs({"units": "1"}) is None


def test_unset_keys_write_nothing() -> None:
    assert ACDD(title="Forest").to_attrs() == {"title": "Forest"}


def test_a_dataset_root_parses_without_a_dtype() -> None:
    ds = xr.Dataset(attrs={"title": "Forest", "license": "CC0"})

    assert attrs.ACDD.from_attrs(ds.attrs) == attrs.ACDD(title="Forest", license="CC0")
