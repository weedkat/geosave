import pytest

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.model.spec import ModelSpec, RasterRequirement, StacRecipe


def coordinate_anchor():
    return GeoAnchor.from_coordinates(
        45,
        12,
        4,
        10,
        crs="EPSG:32633",
        timespan="2025-01",
    )


def recipe():
    return StacRecipe.model_validate(
        {"collection": "optical", "endpoints": ["https://primary.test/stac"]}
    )


def test_load_rasters_is_not_settled():
    model = ModelSpec(
        schema_version=2,
        rasters={"image": RasterRequirement(variables=("red",), stac=recipe())},
    )

    with pytest.raises(NotImplementedError, match="cuts.stacks"):
        model.load_rasters(coordinate_anchor())


def test_a_stac_recipe_does_not_load_rasters_yet():
    with pytest.raises(NotImplementedError, match="cuts.stacks"):
        recipe().load_raster(coordinate_anchor())
