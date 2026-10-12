import matplotlib
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.core.constructors import raster as build_raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.core.geolocator import Place

pytest.importorskip("hvplot", reason="viz extra not installed")

import holoviews as hv  # noqa: E402

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


def panel_axes(element) -> list[matplotlib.axes.Axes]:
    return [axis for axis in hv.render(element).axes if axis.get_xlabel()]


class TestPacking:
    def test_draws_stored_values(self, optical: xr.Dataset) -> None:
        assert np.nanmax(drawn_values(optical.gs.plot("B04"))) == 1500

    def test_leaves_the_declared_fill_in_place(self, optical: xr.Dataset) -> None:
        assert not np.isnan(drawn_values(optical.gs.plot("B04"))).any()

    def test_identity_packing_is_left_alone(self, labels: xr.Dataset) -> None:
        packed = labels.gs.write_packing(1.0, 0.0, target="landcover")
        assert packed.gs.plot("landcover") is not None


class TestClassMap:
    def test_draws_classes_on_position(self, labels: xr.Dataset) -> None:
        assert sorted(set(drawn_values(labels.gs.plot("landcover")))) == [0, 1, 2]

    def test_colours_follow_class_order(self, labels: xr.Dataset) -> None:
        style = (
            labels.gs.plot("landcover").opts.get("style", backend="matplotlib").kwargs
        )
        assert list(style["cmap"].colors) == ["#419bdf", "#397d49", "#b39fe1"]

    def test_names_every_class_on_the_colorbar(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")
        assert drawn.opts.get("plot", backend="matplotlib").kwargs["cbar_ticks"] == [
            (0, "water"),
            (1, "trees"),
            (2, "snow"),
        ]

    def test_titles_no_colorbar_its_ticks_already_name(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")
        assert drawn.opts.get("plot", backend="matplotlib").kwargs["clabel"] == ""

    def test_accepts_an_rgb_tuple_palette(self, labels: xr.Dataset) -> None:
        recoloured = labels.gs.write_legend(
            {0: "water", 1: "trees", 8: "snow"},
            {0: (65, 155, 223), 1: (57, 125, 73), 8: (179, 159, 225)},
            target="landcover",
        )
        style = (
            recoloured.gs.plot("landcover")
            .opts.get("style", backend="matplotlib")
            .kwargs
        )
        assert style["cmap"].colors[0] == "#419bdf"

    def test_absent_pixels_draw_as_no_class(self) -> None:
        codes = np.array(
            [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
        )
        raster = build_raster({"cover": (("y", "x"), codes)}, geobox(), nodata=0)
        raster = raster.gs.write_legend(
            {1: "trees", 8: "snow"}, {1: "#397d49", 8: "#b39fe1"}, target="cover"
        )
        drawn = drawn_values(raster.gs.plot("cover"))
        assert np.isnan(drawn).any()
        assert set(np.unique(drawn[~np.isnan(drawn)])) == {0.0, 1.0}

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
        reflectance = (rgb.astype("float32") / 16 * 0.3).gs.write_rgb("B04", "B03", "B02")

        drawn = reflectance.gs.plot()

        assert float(drawn.data["R"].max()) == 1.0


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

    def test_stack_group_names_join_the_caption(
        self, optical: xr.Dataset, labels: xr.Dataset
    ) -> None:
        scene = build_stack({"optical": optical, "labels": labels})

        axes = panel_axes(scene.gs.plot(cols=2))

        assert [axis.get_title() for axis in axes] == ["", ""]
        assert {axis.get_xlabel().splitlines()[0] for axis in axes} == {
            "labels",
            "optical",
        }


class TestExplore:
    def test_draws_over_web_map_tiles_with_bokeh(self, labels: xr.Dataset) -> None:
        pane = labels.gs.explore()

        assert pane.backend == "bokeh"
        assert [type(layer).__name__ for layer in pane.object] == ["Tiles", "Image"]

    def test_draws_a_copy_on_the_web_map_grid(self, labels: xr.Dataset) -> None:
        image = labels.gs.explore().object.get(1)

        assert labels.gs.crs_name == "EPSG:32749"
        assert image.range("x")[0] > 1e7  # Web Mercator metres, not UTM

    def test_names_classes_on_the_colorbar(self, labels: xr.Dataset) -> None:
        image = labels.gs.explore().object.get(1)

        ticks = hv.Store.lookup_options("bokeh", image, "plot").kwargs["cticks"]
        assert [name for _, name in ticks] == ["water", "trees", "snow"]

    def test_draws_a_band(self, labels: xr.Dataset) -> None:
        assert labels["landcover"].gs.explore().backend == "bokeh"

    def test_draws_the_one_date_of_a_dated_raster(self, rgb: xr.Dataset) -> None:
        dated = rgb.expand_dims(time=[np.datetime64("2024-03-05")])

        assert type(dated.gs.explore().object.get(1)).__name__ == "RGB"

    def test_refuses_several_dates(self, rgb: xr.Dataset) -> None:
        dated = rgb.expand_dims(
            time=[np.datetime64("2024-03-05"), np.datetime64("2024-03-06")]
        )

        with pytest.raises(ValueError, match="isel"):
            dated.gs.explore()

    def test_leaves_static_plots_on_matplotlib(self, labels: xr.Dataset) -> None:
        labels.gs.explore()

        assert hv.Store.current_backend == "matplotlib"
        assert panel_axes(labels.gs.plot())
