import holoviews as hv
import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from odc.geo.geobox import GeoBox

from tests.conftest import build_dataset as build_raster
from geosave.geodata import attrs

matplotlib.use("agg")
hv.extension("matplotlib")


def geobox() -> GeoBox:
    return GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        shape=(4, 4),
        tight=True,
    )


@pytest.fixture
def optical() -> xr.Dataset:
    values = np.arange(16, dtype="uint16").reshape(4, 4) * 100
    raster = build_raster({"B04": (("y", "x"), values)}, geobox())
    return raster.gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4), target=["B04"]
    ).gs.write_nodata(0, target=["B04"])


@pytest.fixture
def labels() -> xr.Dataset:
    """Class map whose class IDs are sparse, as a real land-cover product's are."""
    class_ids = np.array(
        [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
    )
    raster = build_raster({"landcover": (("y", "x"), class_ids)}, geobox())
    return raster.gs.write_flags(
        {0: "water", 1: "trees", 8: "snow"},
        {0: "#419bdf", 1: "#397d49", 8: "#b39fe1"},
        target=["landcover"],
    )


@pytest.fixture
def rgb() -> xr.Dataset:
    pixels = np.arange(16, dtype="uint16").reshape(4, 4)
    raster = build_raster(
        {
            "B04": (("y", "x"), pixels),
            "B03": (("y", "x"), pixels),
            "B02": (("y", "x"), pixels),
        },
        geobox(),
    )
    return raster.gs.write_rgb("B04", "B03", "B02")


def drawn_values(element) -> np.ndarray:
    return element.dimension_values(2)


def panel_axes(element) -> list[Axes]:
    return [axis for axis in hv.render(element).axes if axis.get_xlabel()]


class TestPacking:
    def test_draws_stored_values(self, optical: xr.Dataset) -> None:
        assert np.nanmax(drawn_values(optical.gs.plot("B04"))) == 1500

    def test_leaves_the_declared_fill_in_place(self, optical: xr.Dataset) -> None:
        assert not np.isnan(drawn_values(optical.gs.plot("B04"))).any()


def map_of(drawn) -> hv.RGB:
    """Return the map under a figure's legend."""
    return next(x for x in drawn if isinstance(x, hv.RGB))


class TestClassMap:
    def test_draws_each_class_in_its_colour(self, labels: xr.Dataset) -> None:
        red = map_of(labels.gs.plot("landcover")).dimension_values("R", flat=False)

        # The fixture's top-left pixel is water, #419bdf.
        assert red[0, 0] == pytest.approx(0x41 / 255)

    def test_names_every_class_in_a_legend(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")

        assert [x.label for x in drawn if isinstance(x, hv.Points)] == [
            "water",
            "trees",
            "snow",
        ]
        hv.render(drawn)

    def test_absent_pixels_draw_as_no_class(self) -> None:
        class_ids = np.array(
            [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
        )
        raster = build_raster({"cover": (("y", "x"), class_ids)}, geobox(), nodata=0)
        raster = raster.gs.write_flags(
            {1: "trees", 8: "snow"}, {1: "#397d49", 8: "#b39fe1"}, target=["cover"]
        )

        drawn = map_of(raster.gs.plot("cover"))

        assert np.isnan(drawn.dimension_values("R")).any()


class TestConstraints:
    def test_per_band_colour_interpretation_selects_rgb(self, rgb: xr.Dataset) -> None:
        assert isinstance(rgb.gs.plot(), hv.RGB)

    def test_dark_reflectance_is_stretched_to_fill_the_display(
        self, rgb: xr.Dataset
    ) -> None:
        # Reflectance sits well inside [0, 1]; drawn as-is it would be near black.
        reflectance = (rgb.astype("float32") / 16 * 0.3).gs.write_rgb(
            "B04", "B03", "B02"
        )

        drawn = reflectance.gs.plot()

        assert float(np.nanmax(drawn.dimension_values("R"))) == 1.0


class TestOutput:
    def test_saves_a_png(self, labels: xr.Dataset, tmp_path) -> None:
        written = tmp_path / "landcover.png"
        hv.save(labels.gs.plot("landcover"), written, backend="matplotlib", fmt="png")
        assert written.stat().st_size > 0

    def test_dated_plots_compose(self, rgb: xr.Dataset, labels: xr.Dataset) -> None:
        one = rgb.expand_dims(time=[np.datetime64("2024-03-05")])
        two = rgb.expand_dims(
            time=[np.datetime64("2024-03-05"), np.datetime64("2024-03-06")]
        )

        composed = one.gs.plot() + labels.gs.plot("landcover") + two.gs.plot()

        assert len(composed) == 4
        assert len(panel_axes(composed)) == 4


class TestVariables:
    def test_explicit_band_overrides_recorded_composite(self, rgb: xr.Dataset) -> None:
        plot = rgb.gs.plot("B04", xlabel="")
        assert isinstance(plot, hv.Image)
        np.testing.assert_array_equal(
            np.sort(drawn_values(plot)), np.sort(rgb.B04.values.ravel())
        )

    def test_rgb_draws_a_composite(self, rgb: xr.Dataset) -> None:
        assert isinstance(rgb.gs.plot(rgb=("B04", "B03", "B02")), hv.RGB)

    def test_several_variables_need_a_name_or_a_recorded_composite(
        self, rgb: xr.Dataset
    ) -> None:
        # Dropping the attrs drops the colour interpretation write_rgb recorded.
        with pytest.raises(ValueError, match="records no composite"):
            rgb.drop_attrs().gs.plot()
