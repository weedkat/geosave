import numpy as np
import pytest
from prefect import Flow

from geosave.geodata import attrs, io, open_vector
from tests.conftest import build_dataset
from geosave.model.spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_flows import ingest
from geosave_flows.configs import CoordinateAnchorConfig, RasterAnchorConfig
from tests.geodata.conftest import create_test_collection


def model_spec(tmp_path, url):
    return ModelSpec(
        schema_version=3,
        rasters={
            "optical": RasterRequirement(
                variables=("red", "nir"),
                stac=StacRecipe.model_validate(
                    {
                        "collection": "optical",
                        "endpoints": (url,),
                        "load": {"bands": ["red", "nir"]},
                    }
                ),
                require_crs=True,
            )
        },
    ).save(tmp_path / "model_spec.yaml")


def _dated(geobox, name: str, collection: str) -> object:
    """Serve one dated raster on the requested grid, as a STAC load gives it."""
    plane = np.ones(tuple(geobox.shape), dtype="uint16")
    raster = build_dataset({name: (("y", "x"), plane)}, geobox).expand_dims(
        time=[np.datetime64("2025-01-15")]
    )
    described = create_test_collection(collection, description=f"{collection} source")
    described.links = []
    metadata = attrs.StacMetadata(
        collection=described.to_dict(include_self_link=False, transform_hrefs=False)
    )
    return attrs.stamp_attrs(raster, metadata)


def test_ingest_writes_one_catalog_from_a_raster_anchor(tmp_path, stac_server):
    url, requests, anchor, geobox = stac_server
    label = build_dataset(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))},
        geobox,
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.formats.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "raw"

    result = ingest.fn(
        anchor=RasterAnchorConfig(kind="raster", path=str(label_path)),
        output=str(output),
        spec=str(model_spec(tmp_path, url)),
    )

    assert isinstance(ingest, Flow)
    assert result == str(output / "catalog.parquet")
    assert requests == [["optical"]]
    catalog = open_vector(result)
    assert sorted(catalog["collection"].unique()) == ["optical"]
    assert list(catalog.gs.collections) == ["optical"]
    scene = catalog.iloc[0].gs.to_datatree().gs.to_dataset()
    assert scene.gs.geobox == geobox


def test_ingest_lists_every_model_raster_in_one_catalog(tmp_path, monkeypatch) -> None:
    spec = ModelSpec(
        schema_version=3,
        rasters={
            "optical": RasterRequirement(
                variables=("red",),
                stac=StacRecipe.model_validate(
                    {"collection": "optical", "endpoints": ("https://stac.test",)}
                ),
            ),
            "elevation": RasterRequirement(
                variables=("dem",),
                stac=StacRecipe.model_validate(
                    {"collection": "dem", "endpoints": ("https://stac.test",)}
                ),
            ),
        },
    ).save(tmp_path / "model_spec.yaml")

    def load(model, anchor, *, geobox):
        return {
            "optical": _dated(geobox, "red", "optical"),
            "elevation": _dated(geobox, "dem", "dem"),
        }

    monkeypatch.setattr(ModelSpec, "load_rasters", load)

    result = ingest.fn(
        anchor=CoordinateAnchorConfig(
            kind="coordinates", latitude=45, longitude=12, shape=4, resolution=10
        ),
        output=str(tmp_path / "raw"),
        spec=str(spec),
    )

    catalog = open_vector(result)
    assert sorted(catalog["collection"]) == ["elevation", "optical"]
    # Each raster's source Collection is kept under the raster's own name.
    assert sorted(catalog.gs.collections) == ["elevation", "optical"]
    assert catalog.gs.collections["elevation"].description == "dem source"


def test_ingest_refuses_rasters_without_a_stac_block(tmp_path):
    spec = ModelSpec(
        schema_version=3,
        rasters={
            "optical": RasterRequirement(variables=("red",)),
            "elevation": RasterRequirement(variables=("dem",)),
        },
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(
        ValueError, match=r"no `stac` block: \['optical', 'elevation'\]"
    ):
        ingest.fn(
            anchor=CoordinateAnchorConfig(
                kind="coordinates", latitude=45, longitude=12, shape=4, resolution=10
            ),
            output=str(tmp_path / "raw"),
            spec=str(spec),
        )

    assert not (tmp_path / "raw").exists()


@pytest.mark.slow
def test_ingest_run_opens_a_json_anchor(tmp_path, prefect_server, monkeypatch) -> None:
    spec = ModelSpec(
        schema_version=3,
        rasters={
            "optical": RasterRequirement(
                variables=("red",),
                stac=StacRecipe.model_validate(
                    {"collection": "optical", "endpoints": ("https://stac.test",)}
                ),
            )
        },
    ).save(tmp_path / "model_spec.yaml")
    shapes = []

    def load(model, anchor, *, geobox):
        shapes.append(tuple(geobox.shape))
        return {"optical": _dated(geobox, "red", "optical")}

    monkeypatch.setattr(ModelSpec, "load_rasters", load)

    result = ingest(
        anchor={
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
        },
        output=str(tmp_path / "raw"),
        spec=str(spec),
    )

    assert result == str(tmp_path / "raw" / "catalog.parquet")
    assert shapes == [(4, 4)]


def test_ingest_refuses_an_existing_output(tmp_path, stac_server):
    url, requests, anchor, geobox = stac_server
    label = build_dataset(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))},
        geobox,
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.formats.geotiff.write_cog(label, tmp_path / "label.tif")
    (tmp_path / "raw").mkdir()

    with pytest.raises(FileExistsError):
        ingest.fn(
            anchor=RasterAnchorConfig(kind="raster", path=str(label_path)),
            output=str(tmp_path / "raw"),
            spec=str(model_spec(tmp_path, url)),
        )

    assert requests == []
