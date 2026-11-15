"""Postprocessors the library registers: decode merged model output into a product."""

import xarray as xr

from geosave.ml.processing.roles import POSTPROCESSORS

# Stored value of pixels no chip predicted.
NO_PREDICTION = 255


@POSTPROCESSORS.register("class_map")
class ClassMap:
    """Decode merged logits into a class map that names its classes.

    Args:
        classes: Class names; a class id is its position.
        colors: Hex colour of each class.
    """

    def __init__(self, classes: list[str], colors: list[str]):
        self.classes = classes
        self.colors = colors

    def postprocess(self, logits: xr.DataArray) -> xr.Dataset:
        """Take the strongest class of merged `(band, y, x)` logits.

        Returns:
            Dataset with `class_id` (uint8) on the logits' grid, carrying the
            class names and colours; `NO_PREDICTION` where the logits are
            missing.
        """
        predicted = logits.notnull().all("band")
        class_ids = logits.fillna(0).argmax("band").where(predicted, NO_PREDICTION)
        class_map = class_ids.astype("uint8").to_dataset(name="class_id")

        classes = dict(enumerate(self.classes))
        colors = dict(enumerate(self.colors))
        class_map = class_map.gs.write_flags(classes, colors)
        return class_map.gs.write_nodata(NO_PREDICTION)
