"""Contract for turning chip outputs into a product, the postprocess stage's output, and window stacking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import xarray as xr
from upath import UPath

from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME
from geosave.geodata.utils.datetime import DateRange
from geosave.ml.pipeline.prediction import PredictOutput


@dataclass(frozen=True)
class PostprocessOutput[Product]:
    """What the postprocess stage hands a publisher: the task's product.

    Attributes:
        product: A raster Dataset for dense tasks, a GeoDataFrame for
            detections; lazy over the run's store when one is given.
        metadata: Values a custom stage adds for a later one.
    """

    product: Product
    metadata: dict[str, Any] = field(default_factory=dict)


class Postprocessor[Output, Product](ABC):
    """A maker of one task's product from a model's chip outputs.

    It reads each chip's raw output where its location places it, puts the
    chips of each window together on the window's grid, which the chipper
    states, and decodes what the outputs mean for its task: a raster for
    dense tasks, a GeoDataFrame for detections. A product too large for
    memory is staged under the store it is given.
    """

    @abstractmethod
    def __call__(
        self, predicted: PredictOutput[Output], store: UPath | None = None
    ) -> PostprocessOutput[Product]:
        """Return the product of every window the chips were cut from.

        Args:
            predicted: The predict stage's output.
            store: Folder to stage the product in; None holds it in memory.
        """


def stack_windows(
    products: list[xr.Dataset], timespans: list[DateRange | None]
) -> xr.Dataset:
    """Stack each window's raster product along `time`, dated by its window.

    A window's label is the start of its timespan, and `start_datetime` and
    `end_datetime` state the whole span. A pipeline without temporal windows
    predicts one timeless window, whose product is returned without `time`.

    Args:
        products: One product per window, in window order.
        timespans: Each window's timespan, as its chips' source states it.

    Returns:
        The products along `time`, or the one timeless product.

    Raises:
        ValueError: No window was predicted, or a timeless window is among
            several.
    """
    if not products:
        raise ValueError("no window was predicted")
    if timespans == [None]:
        return products[0]

    dated: list[xr.Dataset] = []
    for product, timespan in zip(products, timespans, strict=True):
        if timespan is None:
            raise ValueError("a timeless window cannot be stacked with others")
        start, end = pd.Timestamp(timespan[0]), pd.Timestamp(timespan[1])
        product = product.expand_dims(time=[start])
        product = product.assign_coords(
            {START_DATETIME: ("time", [start]), END_DATETIME: ("time", [end])}
        )
        dated.append(product)
    return xr.concat(dated, "time")
