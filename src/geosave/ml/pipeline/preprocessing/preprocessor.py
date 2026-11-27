"""Turn the ingested sources into model inputs, one per temporal window."""

from dataclasses import dataclass, field
from typing import Any, Literal

import xarray as xr

from geosave.geodata.sampling import iter_samples
from geosave.geodata.utils.datetime import Freq
from geosave.ml.pipeline.ingestion import IngestOutput

from .steps import Step


@dataclass(frozen=True)
class PreprocessOutput:
    """What the preprocess stage hands the predictor: the model inputs.

    Attributes:
        windows: One prepared tree per temporal window, still lazy; its
            raster node paths name the model's inputs.
        metadata: Values a custom stage adds for a later one.

    Raises:
        ValueError: There is no window to predict.
    """

    windows: tuple[xr.DataTree, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("the preprocessor produced no complete temporal window")


class Preprocessor:
    """Run steps in order on the ingested tree, then split it into temporal windows.

    Each step turns a tree into a tree, so their order is the recipe: decode
    before compositing, mask clouds before reducing over time. A window keeps
    one observation per time bucket in every dated node. Each window is one
    model input on its own grid, still lazy.

    Args:
        steps: Operations in the order they run.
        time_freq: Time bucket length, such as "MS" for monthly buckets.
            None yields the prepared tree as the one input.
        time_window: Number of consecutive buckets in a window.
        time_stride: Number of buckets between window starts.
        time_keep: Which observation a bucket keeps when several occupy it.
        time_missing: `"skip"` omits windows with empty buckets. `"raise"`
            rejects the prepared tree before yielding any window.

    Examples:
        >>> preprocessor = Preprocessor(
        ...     [MaskAndScale(), Composite("MS", "median")], time_freq="MS"
        ... )
        >>> windows = preprocessor(ingested).windows
    """

    def __init__(
        self,
        steps: list[Step],
        *,
        time_freq: Freq | None = None,
        time_window: int = 1,
        time_stride: int = 1,
        time_keep: Literal["first", "last"] = "last",
        time_missing: Literal["skip", "raise"] = "skip",
    ) -> None:
        self.steps = steps
        self.time_freq = time_freq
        self.time_window = time_window
        self.time_stride = time_stride
        self.time_keep: Literal["first", "last"] = time_keep
        self.time_missing: Literal["skip", "raise"] = time_missing

    def __call__(self, ingested: IngestOutput) -> PreprocessOutput:
        """Return each window of the prepared tree.

        Args:
            ingested: The ingest stage's output.

        Returns:
            The prepared tree with each dated node on each window's
            observations, or the whole prepared tree when `time_freq` is None.

        Raises:
            ValueError: A step refuses the tree, no window is complete, or a
                window has an empty bucket under `time_missing="raise"`.
        """
        prepared = ingested.sources
        for step in self.steps:
            prepared = step(prepared)

        if self.time_freq is None:
            return PreprocessOutput((prepared,))

        windows = iter_samples(
            prepared,
            self.time_freq,
            size=self.time_window,
            stride=self.time_stride,
            keep=self.time_keep,
            missing=self.time_missing,
        )
        return PreprocessOutput(tuple(windows))
