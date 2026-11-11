"""Items read back as the files and variable attrs they describe."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pystac
import pytest
from pystac.extensions.eo import EOExtension

from geosave.geodata import attrs, stac
from geosave.geodata.stac.load import load_datatree
from geosave.geodata.stac import band
from geosave.geodata.stac.catalog import create_catalog
from tests.geodata.conftest import build_raster, create_file_items


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    files = scene.gs.to_cog(
        tmp_path, format_paths="forest_{time:%Y%m%d}_{variable}.tif"
    )
    entry = create_catalog(
        [next(create_file_items(files, id_format="forest"))]
    ).gs.to_items()[0]

    back = load_datatree([entry]).gs.to_dataset()
    assert sorted(back.gs.variables) == ["nir", "red"]
    assert back.gs.times[0] == pd.Timestamp("2025-06-01")


def test_a_masked_raster_publishes_its_mask_as_footprint_and_its_full_grid(tmp_path):
    from odc.geo.geom import Geometry
    from pystac.extensions.projection import ProjectionExtension
    from shapely.geometry import Polygon, shape

    raster = build_raster(times=1)
    grid = raster.gs.geobox
    # Pixels outside the triangle are nodata, so it is where the data is.
    mask = Geometry(Polygon(list(grid.extent.geom.exterior.coords)[:3]), grid.crs)
    masked = raster.gs.write_extent(mask)

    entry = masked.gs.to_item(tmp_path / "published", split_asset=())

    assert shape(entry.geometry).equals_exact(mask.to_crs("EPSG:4326").geom, 1e-9)
    for asset in entry.assets.values():
        projection = ProjectionExtension.ext(asset)
        assert tuple(projection.shape) == grid.shape
        assert tuple(projection.transform[:6]) == tuple(grid.transform)[:6]


def _item(tmp_path, assets):
    """Build an Item whose data assets list EO bands, as `(key, bands)` pairs."""
    from datetime import UTC, datetime

    from pystac.extensions.eo import Band

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    EOExtension.add_to(item)
    for key, bands in assets:
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = [Band.create(**b) for b in bands]
    return item


def test_a_rendering_of_other_assets_bands_names_them_by_key(tmp_path) -> None:
    # Planetary Computer's shape: per-band files beside a true-colour rendering
    # whose eo:bands repeat their names, with no raster:bands count.
    item = _item(
        tmp_path,
        [
            ("B04", [{"name": "B04"}]),
            ("B03", [{"name": "B03"}]),
            ("B02", [{"name": "B02"}]),
            ("visual", [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}]),
        ],
    )

    files = stac.item.read_variable_names(item)

    assert list(files.values()) == [
        ("B04",),
        ("B03",),
        ("B02",),
        ("visual.1", "visual.2", "visual.3"),
    ]


def test_several_bands_of_one_file_take_the_names_they_state(tmp_path) -> None:
    # As odc-stac names them: the EO name, else the common name, else key.i.
    item = _item(
        tmp_path,
        [
            (
                "image",
                [{"name": "B02"}, {"name": None, "common_name": "nir"}, {"name": None}],
            )
        ],
    )

    (names,) = stac.item.read_variable_names(item).values()

    assert names == ("B02", "nir", "image.3")


def test_a_name_claimed_by_a_one_band_asset_falls_back_to_key(tmp_path) -> None:
    item = _item(
        tmp_path,
        [
            ("rgb", [{"name": "red"}, {"name": "green"}]),
            ("red", [{"name": "red"}]),
        ],
    )

    files = stac.item.read_variable_names(item)

    assert list(files.values()) == [
        ("rgb.1", "green"),
        ("red",),
    ]


def test_attrs_carry_band_facts_by_variable_name(tmp_path) -> None:
    from datetime import UTC, datetime

    from pystac.extensions.eo import Band

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    EOExtension.add_to(item)
    listed = {
        "B04": [Band.create(name="B04", common_name="red", center_wavelength=0.665)],
        "visual": [Band.create(name=n) for n in ("B04", "B03", "B02")],
    }
    for key, bands in listed.items():
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = bands

    red = band.read_attrs(item.assets["B04"], 0)
    visual = band.read_attrs(item.assets["visual"], 0)
    assert attrs.Spectral.from_attrs(red).common_name == "red"
    assert visual == {}


def test_composite_keeps_nominal_time_and_bounds(tmp_path):

    # A monthly composite covers its June bin.
    ds = build_raster(times=2).gs.mask_and_scale().gs.composite("MS", "median")
    scene = ds.gs.to_cog(tmp_path, format_paths="composite.tif", split_asset=())[0]
    item = next(create_file_items(scene))
    assert item.datetime is None
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-07-01")


def test_unsplit_multidate_store_cannot_be_one_observation(tmp_path):
    ds = build_raster(times=2)
    with pytest.raises(ValueError, match="one observation"):
        ds.gs.to_item(tmp_path)
    split = ds.gs.to_cog(
        tmp_path, format_paths="date-{time:%Y%m%d}.tif", split_asset=()
    )
    assert len(list(create_file_items(split))) == 2


def test_coordinates_are_not_properties_by_their_name(tmp_path):
    ds = build_raster(times=1).assign_coords({"eo:cloud_cover": 12.0, "platform": "x"})
    item = next(
        create_file_items(
            ds.gs.to_cog(tmp_path, format_paths="named.tif", split_asset=())
        )
    )
    assert "eo:cloud_cover" not in item.properties
    assert "platform" not in item.properties


def test_observation_properties_leave_out_what_a_raster_states():
    kept = stac.item.select_item_properties(
        {
            "datetime": "2025-01-15T00:00:00Z",
            "proj:shape": [10, 10],
            "eo:bands": [],
            "eo:cloud_cover": 4.2,
            "platform": "sentinel-2a",
        }
    )
    assert kept == {"eo:cloud_cover": 4.2, "platform": "sentinel-2a"}


def test_equal_explicit_interval_is_an_instant(tmp_path):
    ds = build_raster()
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    item = next(
        create_file_items(
            ds.gs.to_cog(tmp_path, format_paths="undated.tif", split_asset=()),
            datetime=(stamp, stamp),
        )
    )
    assert item.datetime == stamp


def test_normalized_time_preserves_nominal_date_and_interval():
    from geosave.geodata.stac.item import create_item, normalize_time

    time = normalize_time(
        "2025-06-15T02:00:00+02:00",
        "2025-06-01",
        "2025-07-01",
    )
    properties = {"platform": "sentinel-2a", "instruments": ["msi"]}
    entry = create_item(id="composite", time=time, properties=properties)

    assert entry.datetime == datetime(2025, 6, 15, tzinfo=UTC)
    assert entry.properties["start_datetime"] == "2025-06-01T00:00:00+00:00"
    assert entry.properties["end_datetime"] == "2025-07-01T00:00:00+00:00"
    entry.properties["instruments"].append("other")
    assert properties == {"platform": "sentinel-2a", "instruments": ["msi"]}


@pytest.mark.parametrize(
    ("instant", "start", "end"),
    [
        (None, None, None),
        ("NaT", None, None),
        (None, "2025-06-01", None),
        (None, "2025-07-01", "2025-06-01"),
        ("2025-08-01", "2025-06-01", "2025-07-01"),
    ],
)
def test_invalid_observation_time_is_refused(instant, start, end):
    from geosave.geodata.stac.item import normalize_time

    with pytest.raises(ValueError):
        normalize_time(instant, start, end)


def test_an_item_period_is_its_stated_interval_or_its_instant():
    from datetime import UTC, datetime

    from geosave.geodata.stac.item import TIMELESS, read_item_period

    scene = pystac.Item("scene", None, None, datetime(2025, 6, 1, 10, tzinfo=UTC), {})
    composite = pystac.Item(
        "composite",
        None,
        None,
        None,
        {
            "start_datetime": "2025-06-01T00:00:00Z",
            "end_datetime": "2025-07-01T00:00:00Z",
        },
    )
    timeless = pystac.Item(
        "timeless",
        None,
        None,
        None,
        {
            "start_datetime": TIMELESS[0].isoformat(),
            "end_datetime": TIMELESS[1].isoformat(),
        },
    )

    assert read_item_period(scene) == (
        pd.Timestamp("2025-06-01T10:00"),
        pd.Timestamp("2025-06-01T10:00"),
    )
    assert read_item_period(composite) == (
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-07-01"),
    )
    assert read_item_period(timeless) is None
