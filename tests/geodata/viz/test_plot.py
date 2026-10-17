import geopandas as gpd
import holoviews as hv
import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from odc.geo.geobox import GeoBox
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon

from geosave.geodata.core.anchor import GeoAnchor
from geosave.geodata.core.raster.create import raster as build_raster
from geosave.geodata.core.raster.create import tree as build_tree
from geosave.geodata.utils.geocode import Place

matplotlib.use("agg")
hv.extension("matplotlib")


@pytest.fixture(autouse=True)
def resolved_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GeoAnchor, "locate", lambda _: Place(city="Test City"))


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
    return raster.gs.write_packing(1e-4, target="B04").gs.write_nodata(0, target="B04")


@pytest.fixture
def labels() -> xr.Dataset:
    """Class map whose codes are sparse, as a real land-cover product's are."""
    codes = np.array(
        [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
    )
    raster = build_raster({"landcover": (("y", "x"), codes)}, geobox())
    return raster.gs.write_legend(
        {0: "water", 1: "trees", 8: "snow"},
        {0: "#419bdf", 1: "#397d49", 8: "#b39fe1"},
        target="landcover",
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

    def test_identity_packing_is_left_alone(self, labels: xr.Dataset) -> None:
        packed = labels.gs.write_packing(1.0, 0.0, target="landcover")
        assert packed.gs.plot("landcover") is not None


def map_of(drawn) -> hv.RGB:
    """Return the map under a figure's legend."""
    return next(x for x in drawn if isinstance(x, hv.RGB))


class TestClassMap:
    def test_composed_maps_share_one_legend_on_the_figure_right(self, labels):
        drawn = labels.gs.plot("landcover") + labels.gs.plot("landcover")

        figure = hv.render(drawn)
        figure.canvas.draw()

        assert len(figure.legends) == 1
        legend = figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "water",
            "trees",
            "snow",
        ]
        bounds = legend.get_window_extent()
        for axis in figure.axes:
            assert axis.get_legend() is None
            assert bounds.x0 > axis.get_window_extent().x1

    def test_dated_maps_share_one_legend(self, labels):
        dated = labels.expand_dims(
            time=np.array(["2025-01-01", "2025-02-01"], "datetime64[D]")
        )

        figure = hv.render(dated.gs.plot("landcover", cols=2))

        assert len(figure.legends) == 1
        assert len(figure.legends[0].get_texts()) == 3

    def test_shared_legend_keeps_caller_styling(self, labels):
        drawn = labels.gs.plot("landcover").opts(
            legend_opts={"fontsize": 18, "frameon": False}
        )

        figure = hv.render(drawn)

        assert not figure.legends[0].get_frame_on()
        assert figure.legends[0].get_texts()[0].get_fontsize() == 18

    def test_same_name_with_different_colour_remains_in_the_legend(self, labels):
        changed = labels.gs.write_legend(
            {0: "water", 1: "trees", 8: "snow"},
            {0: "#0000ff", 1: "#397d49", 8: "#b39fe1"},
            target="landcover",
        )

        figure = hv.render(labels.gs.plot("landcover") + changed.gs.plot("landcover"))

        assert len(figure.legends) == 1
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "water",
            "trees",
            "snow",
            "water",
        ]

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

    def test_accepts_an_rgb_tuple_palette(self, labels: xr.Dataset) -> None:
        recoloured = labels.gs.write_legend(
            {0: "water", 1: "trees", 8: "snow"},
            {0: (65, 155, 223), 1: (57, 125, 73), 8: (179, 159, 225)},
            target="landcover",
        )

        red = map_of(recoloured.gs.plot("landcover")).dimension_values("R", flat=False)

        assert red[0, 0] == pytest.approx(0x41 / 255)

    def test_absent_pixels_draw_as_no_class(self) -> None:
        codes = np.array(
            [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
        )
        raster = build_raster({"cover": (("y", "x"), codes)}, geobox(), nodata=0)
        raster = raster.gs.write_legend(
            {1: "trees", 8: "snow"}, {1: "#397d49", 8: "#b39fe1"}, target="cover"
        )

        drawn = map_of(raster.gs.plot("cover"))

        assert np.isnan(drawn.dimension_values("R")).any()

    def test_refuses_a_colormap_or_limits(self, labels: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="drop cmap"):
            labels.gs.plot("landcover", cmap="gray")
        with pytest.raises(ValueError, match="clim cannot apply"):
            labels.gs.plot("landcover", clim=(0, 1))

    def test_uncoloured_class_refuses(self) -> None:
        raster = build_raster(
            {"cover": (("y", "x"), np.zeros((4, 4), dtype="uint8"))}, geobox()
        )
        raster = raster.gs.write_legend(
            {0: "water", 1: "trees"}, {0: "#419bdf"}, target="cover"
        )
        with pytest.raises(ValueError, match="carry no colour"):
            raster.gs.plot("cover")


class TestConstraints:
    def test_unknown_variable_refuses(self, optical: xr.Dataset) -> None:
        with pytest.raises(KeyError, match="not data variables"):
            optical.gs.plot("B99")

    def test_selecting_first_draws(self, optical: xr.Dataset) -> None:
        assert optical.expand_dims(time=3).isel(time=0).gs.plot("B04") is not None

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

    def test_opts_reach_hvplot(self, optical: xr.Dataset) -> None:
        drawn = optical.gs.plot("B04", cmap="RdYlGn")
        assert drawn.opts.get("style").kwargs["cmap"] == "RdYlGn"

    def test_elements_compose(self, optical: xr.Dataset, labels: xr.Dataset) -> None:
        assert isinstance(
            optical.gs.plot("B04") + labels.gs.plot("landcover"), hv.Layout
        )

    def test_dated_plots_compose(self, rgb: xr.Dataset, labels: xr.Dataset) -> None:
        one = rgb.expand_dims(time=[np.datetime64("2024-03-05")])
        two = rgb.expand_dims(
            time=[np.datetime64("2024-03-05"), np.datetime64("2024-03-06")]
        )

        composed = one.gs.plot() + labels.gs.plot("landcover") + two.gs.plot()

        assert len(composed) == 4
        assert len(panel_axes(composed)) == 4

    def test_time_and_location_caption_each_panel(self, rgb: xr.Dataset) -> None:
        dated = rgb.expand_dims(
            time=np.array(
                ["2026-07-02T02:55:19", "2026-07-09T02:45:29"], "datetime64[s]"
            )
        )

        axes = panel_axes(dated.gs.plot(cols=2))

        assert [axis.get_title() for axis in axes] == ["", ""]
        assert axes[0].get_xlabel().startswith("2026-07-02 02:55:19\nTest City")
        assert axes[1].get_xlabel().startswith("2026-07-09 02:45:29\nTest City")

    def test_explicit_title_stays_above_the_axes(self, optical: xr.Dataset) -> None:
        axis = panel_axes(optical.gs.plot("B04", title="Optical"))[0]

        assert axis.get_title() == "Optical"
        assert axis.get_xlabel().startswith("Test City")

    def test_tree_group_names_join_the_caption(
        self, optical: xr.Dataset, labels: xr.Dataset
    ) -> None:
        scene = build_tree({"optical": optical, "labels": labels})

        axes = panel_axes(scene.gs.plot(cols=2))

        assert [axis.get_title() for axis in axes] == ["", ""]
        assert {axis.get_xlabel().splitlines()[0] for axis in axes} == {
            "labels",
            "optical",
        }


class TestVariables:
    def test_rgb_draws_a_composite(self, rgb: xr.Dataset) -> None:
        assert isinstance(rgb.gs.plot(rgb=("B04", "B03", "B02")), hv.RGB)

    def test_names_a_band_or_a_composite_not_both(self, rgb: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="not both"):
            rgb.gs.plot("B04", rgb=("B04", "B03", "B02"))

    def test_a_composite_refuses_a_colormap(self, rgb: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="drop cmap"):
            rgb.gs.plot(cmap="gray")

    def test_several_variables_need_a_name_or_a_recorded_composite(
        self, rgb: xr.Dataset
    ) -> None:
        # Dropping the attrs drops the colour interpretation write_rgb recorded.
        with pytest.raises(ValueError, match="records no composite"):
            rgb.drop_attrs().gs.plot()

    def test_rgb_names_the_composite(self, rgb: xr.Dataset) -> None:
        assert rgb.gs.rgb == ("B04", "B03", "B02")
        assert rgb[["B04"]].gs.rgb is None


RING = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(2, 2), (4, 2), (4, 4), (2, 4)]


def frame(*shapes, crs="EPSG:32749") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(shapes), crs=crs)


class TestOutline:
    def test_draws_polygons_with_holes(self) -> None:
        drawn = frame(Polygon(RING, [HOLE])).gs.plot()

        (polygons,) = drawn
        assert isinstance(polygons, hv.Polygons)
        assert [len(holes) for holes in polygons.holes()] == [1]
        assert polygons.range("x") == (0, 10)
        hv.render(drawn)

    def test_draws_lines_and_points(self) -> None:
        (lines,) = frame(LineString(RING)).gs.plot()
        (points,) = frame(MultiPoint([(1, 1), (2, 2)])).gs.plot()

        assert isinstance(lines, hv.Path) and isinstance(points, hv.Points)

    def test_overlays_mixed_geometry(self) -> None:
        drawn = frame(Polygon(RING), Point(20, 20)).gs.plot()

        assert [type(layer).__name__ for layer in drawn] == ["Polygons", "Points"]
        hv.render(drawn)

    def test_refuses_nothing_to_draw_and_collections(self) -> None:
        with pytest.raises(ValueError, match="no geometry"):
            frame().gs.plot()
        with pytest.raises(ValueError, match="explode"):
            frame(GeometryCollection([Point(0, 0)])).gs.plot()
