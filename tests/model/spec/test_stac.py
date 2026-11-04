import pytest
from pydantic import ValidationError
from pystac_client.exceptions import APIError

from geosave.geodata.errors import CollectionNotFoundError
from geosave.model.spec import RasterRequirement, StacRecipe


def test_named_variables_must_match_explicit_stac_bands():
    recipe = {
        "collection": "sentinel-2-l2a",
        "endpoints": ["https://primary.test/stac"],
        "load": {"bands": ["B08", "B04"]},
    }

    with pytest.raises(ValidationError, match="bands.*variables"):
        RasterRequirement.model_validate({"variables": ("B04", "B08"), "stac": recipe})

    requirement = RasterRequirement.model_validate(
        {"variables": ("B08", "B04"), "stac": recipe}
    )
    assert requirement.stac is not None


def test_a_recipe_loads_with_its_own_query_and_load_settings(
    serve, red, coordinate_anchor, coordinate_grid
):
    serve({"https://primary.test/stac": red})
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": ["https://primary.test/stac"],
            "query": {"max_items": 3},
            "load": {"bands": ["red"]},
        }
    )

    loaded = recipe.load_raster(coordinate_anchor, geobox=coordinate_grid)

    assert loaded.attrs["config"].bands == ["red"]
    assert loaded.attrs["query"].collections == ["optical"]
    assert loaded.attrs["query"].max_items == 3


def test_a_recipe_keeps_the_signer_its_client_loads_with(
    serve, red, coordinate_anchor, coordinate_grid
):
    def sign(url):
        return f"{url}?token"

    serve({"planetary_computer": red}, patch_url=sign)
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": ["planetary_computer"],
            "load": {"bands": ["red"]},
        }
    )

    loaded = recipe.load_raster(coordinate_anchor, geobox=coordinate_grid)

    assert loaded.attrs["config"].patch_url is sign
    assert loaded.attrs["config"].bands == ["red"]


def test_an_unavailable_endpoint_falls_back_to_the_next(
    serve, red, coordinate_anchor, coordinate_grid
):
    opened = serve(
        {
            "https://primary.test/stac": APIError("unreachable"),
            "https://backup.test/stac": CollectionNotFoundError("absent"),
            "planetary_computer": red,
        },
    )
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": [
                "https://primary.test/stac",
                "https://backup.test/stac",
                "planetary_computer",
            ],
        }
    )

    assert "red" in recipe.load_raster(coordinate_anchor, geobox=coordinate_grid)
    assert opened == [
        "https://primary.test/stac",
        "https://backup.test/stac",
        "planetary_computer",
    ]


def test_no_serving_endpoint_names_every_failure(
    serve, coordinate_anchor, coordinate_grid, stac_recipe
):
    serve({"https://primary.test/stac": APIError("unreachable")})

    with pytest.raises(ConnectionError, match="primary.test.*unreachable"):
        stac_recipe.load_raster(coordinate_anchor, geobox=coordinate_grid)


@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"bbox": [0, 0, 1, 1], "datetime": "2020"},
        {"ids": ["scene"]},
    ],
)
def test_recipe_preserves_selector_defaulting_and_id_bypass(
    serve, red, coordinate_anchor, coordinate_grid, selectors
):
    serve({"planetary_computer": red})
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": ["planetary_computer"],
            "query": selectors,
        }
    )
    loaded = recipe.load_raster(coordinate_anchor, geobox=coordinate_grid)
    query = loaded.attrs["query"]
    if "ids" in selectors:
        assert list(query.ids) == selectors["ids"]
        assert (
            query.bbox is None and query.intersects is None and query.datetime is None
        )
    elif selectors:
        assert query.bbox == tuple(selectors["bbox"])
        assert query.datetime == selectors["datetime"]
    else:
        assert query.bbox == tuple(
            coordinate_anchor.geometry.to_crs("EPSG:4326").boundingbox
        )
        assert query.datetime == coordinate_anchor.timespan
