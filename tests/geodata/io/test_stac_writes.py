"""Publish raster observations as catalog records with their carried metadata."""

import pytest

from tests.geodata.conftest import create_test_collection


def _loaded(collection_id: str = "s2"):
    """Return a two-date raster carrying STAC metadata, as load_dataset records it."""
    from geosave.geodata import attrs
    from geosave.geodata.stac.item import stamp_properties_coordinate
    from tests.geodata.conftest import build_raster

    described = create_test_collection(collection_id, description="Reflectance")
    described.links = []
    raster = attrs.stamp_attrs(
        build_raster(times=2),
        attrs.StacMetadata(
            collection=described.to_dict(
                include_self_link=False, transform_hrefs=False
            ),
            common_properties={"constellation": "sentinel-2"},
        ),
    )
    return stamp_properties_coordinate(
        raster, [{"eo:cloud_cover": 20.0}, {"eo:cloud_cover": 40.0}]
    )


def test_a_loaded_raster_catalogs_under_the_collection_it_carries(tmp_path):
    catalog = _loaded().gs.to_catalog(tmp_path, path_format="{time:%Y%m%d}.tif")

    assert catalog["collection"].tolist() == ["s2", "s2"]
    assert catalog["eo:cloud_cover"].tolist() == [20.0, 40.0]
    assert catalog["constellation"].tolist() == ["sentinel-2", "sentinel-2"]
    # The carried Collection's extent is set to cover the written Items.
    (interval,) = catalog.gs.collections["s2"].extent.temporal.intervals
    assert [instant.day for instant in interval] == [1, 2]


def test_a_derived_product_names_its_own_collection(tmp_path):
    catalog = _loaded().gs.to_catalog(
        tmp_path, path_format="{time:%Y%m%d}.tif", collection="ndvi"
    )

    assert catalog["collection"].unique().tolist() == ["ndvi"]
    # An id alone states no Collection, and the carried one is another product.
    assert catalog.gs.collections == {}


def test_a_tree_keeps_each_groups_collection_under_the_group_name(tmp_path):
    from tests.conftest import build_tree

    tree = build_tree({"optical": _loaded("sentinel-2-l2a")})
    catalog = tree.gs.to_catalog(tmp_path, path_format="{collection}_{time:%Y%m%d}.tif")

    assert list(catalog.gs.collections) == ["optical"]
    assert catalog.gs.collections["optical"].description == "Reflectance"


def test_a_selection_keeping_no_dates_is_refused_before_writing(tmp_path):
    import pytest

    empty = _loaded().isel(time=slice(0, 0))

    with pytest.raises(ValueError, match="nonempty"):
        empty.gs.to_catalog(tmp_path, path_format="{time:%Y%m%d}.tif")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_an_invalid_item_id_leaves_assets_untouched(tmp_path, existing):
    path = tmp_path / "scene.tif"
    if existing:
        path.write_bytes(b"original")

    with pytest.raises(ValueError, match="unknown"):
        _loaded().isel(time=0).gs.to_catalog(
            path.parent,
            path_format=path.name,
            id_format="{unknown}",
            overwrite=existing,
        )

    if existing:
        assert path.read_bytes() == b"original"
    else:
        assert not path.exists()


def test_duplicate_item_ids_are_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="duplicate Item identity"):
        _loaded().gs.to_catalog(
            tmp_path, path_format="{time:%Y%m%d}.tif", id_format="same"
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_incompatible_item_properties_leave_assets_untouched(tmp_path, existing):
    import pyarrow as pa

    from geosave.geodata.stac.item import stamp_properties_coordinate

    paths = [tmp_path / "20250601.tif", tmp_path / "20250602.tif"]
    if existing:
        for path in paths:
            path.write_bytes(b"original")
    raster = stamp_properties_coordinate(
        _loaded(), [{"custom": 1}, {"custom": {"x": 2}}]
    )

    with pytest.raises(pa.ArrowInvalid, match="struct"):
        raster.gs.to_catalog(
            tmp_path, path_format="{time:%Y%m%d}.tif", overwrite=existing
        )

    for path in paths:
        if existing:
            assert path.read_bytes() == b"original"
        else:
            assert not path.exists()


def test_catalog_references_the_file_returned_by_the_writer(tmp_path, monkeypatch):
    import numpy as np

    from geosave.geodata.io.formats import geotiff

    write = geotiff.write_cog

    def save(raster, destination, **options):
        return write(raster, destination.with_name("saved.tif"), **options)

    monkeypatch.setattr(geotiff, "write_cog", save)
    raster = _loaded().isel(time=0)
    catalog = raster.gs.to_catalog(tmp_path, path_format="planned.tif")

    assert not (tmp_path / "planned.tif").exists()
    assert catalog.iloc[0].gs.to_item().assets["image"].href == str(
        tmp_path / "saved.tif"
    )
    with catalog.iloc[0].gs.to_datatree() as tree:
        restored = tree.gs.to_dataset()
        np.testing.assert_array_equal(restored.red.values, raster.red.values)
        assert restored.gs.geobox == raster.gs.geobox


def test_a_mixed_grid_tree_publishes_one_item_per_date(tmp_path):
    from tests.geodata.core.raster.test_datatree import build_mixed_collection

    tree = build_mixed_collection()

    rows = tree.gs.to_catalog(
        tmp_path, path_format="{collection}/{time:%Y%m%d}/{band}.tif"
    )

    assert len(rows) == 2
    item = rows.iloc[0].gs.to_item()
    assert set(item.assets) == {"red", "swir"}
    shapes = {
        key: tuple(asset.extra_fields["proj:shape"])
        for key, asset in item.assets.items()
    }
    assert shapes == {"red": (64, 64), "swir": (32, 32)}


def test_publication_round_trips_through_to_datatree(tmp_path):
    import numpy as np

    from tests.geodata.core.raster.test_datatree import build_mixed_collection

    tree = build_mixed_collection()

    rows = tree.gs.to_catalog(
        tmp_path, path_format="{collection}/{time:%Y%m%d}/{band}.tif"
    )

    with rows.gs.to_datatree() as restored:
        assert list(restored.gs.rasters) == list(tree.gs.rasters)
        for path, raster in tree.gs.rasters.items():
            loaded = restored.gs.rasters[path]
            assert loaded.gs.geobox == raster.gs.geobox
            for name in raster.data_vars:
                np.testing.assert_array_equal(loaded[name].values, raster[name].values)


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_a_store_template_is_refused_for_catalogs(tmp_path, suffix):
    from tests.geodata.conftest import build_raster

    with pytest.raises(ValueError, match="to_zarr or to_netcdf"):
        build_raster(times=1).gs.to_catalog(
            tmp_path, path_format="{time:%Y%m%d}" + suffix, collection="s2"
        )
    assert not list(tmp_path.iterdir())


def test_several_multi_band_files_of_one_date_point_to_band(tmp_path):
    from tests.conftest import build_tree
    from tests.geodata.core.raster.test_datatree import build_mixed_collection

    nodes = {}
    for path, raster in build_mixed_collection().gs.rasters.items():
        name = next(iter(raster.data_vars))
        nodes[path] = raster.assign({f"{name}_copy": raster[name]})
    tree = build_tree(nodes)

    with pytest.raises(ValueError, match=r"\{band\}"):
        tree.gs.to_catalog(tmp_path, path_format="{collection}/{res}/{time:%Y%m%d}.tif")
    assert not list(tmp_path.iterdir())
