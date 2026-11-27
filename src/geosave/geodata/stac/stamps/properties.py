"""Move Item properties loaded by ODC onto the output timeline."""

from collections.abc import Mapping, Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig
from odc.stac.model import PropertyLoadRequest


def stamp_properties(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
    *,
    names: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Mark loaded property variables as coordinates with their original STAC key.

    Request the values with ODC's `with_properties` option when loading.
    ODC has already grouped the values with the pixels. For custom output
    names, pass a mapping such as `names={"cloud": "eo:cloud_cover"}`.
    """
    property_keys = {}

    for item in items:
        for key in item.properties:
            name = PropertyLoadRequest(key).output_name
            if name in ds.data_vars and ds[name].dims == ("time",):
                property_keys[name] = key

    if names:
        property_keys.update(names)

    coordinates = {}
    for name, key in property_keys.items():
        marked = ds[name].assign_attrs(stac_property=key)
        coordinates[name] = marked.variable

    return ds.assign_coords(coordinates)
