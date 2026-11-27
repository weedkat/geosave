"""The contract of a module in a `ModelChain`, and the tensor types modules pass.

A chain keeps one record of named values. A module reads its forward
parameters from it by name and writes its returned values under the names in
its `returns`. The chain checks each read against the record's type hint
when it is built, so annotate a custom module's forward with these types to
connect it to the library's modules.

Examples:
    >>> from geosave.ml.torch.typing import FeatureMap
    >>> class Decoder(nn.Module):
    ...     returns = ("feature_map",)
    ...     def forward(self, pyramid: list[FeatureMap]) -> FeatureMap: ...
"""

from typing import Annotated, Any, ClassVar, Protocol

import torch


class ChainModule(Protocol):
    """A module a `ModelChain` runs.

    Its forward's parameters, all required, are the names it reads. `returns`
    names what forward returns: one name per member of a `tuple[...]` return
    hint, or one name for any other hint.

    Attributes:
        returns: The names of the values forward returns, in order.

    Examples:
        >>> class Unet(nn.Module):
        ...     returns = ("feature_map",)
        ...     def forward(self, pyramid: list[FeatureMap]) -> FeatureMap: ...
    """

    returns: ClassVar[tuple[str, ...]]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Return the values `returns` names."""
        ...


Image = Annotated[torch.Tensor, "image (batch, band, y, x)"]
"""Prepared pixels a model reads, one band per channel."""

ImageSeries = Annotated[torch.Tensor, "image series (batch, time, band, y, x)"]
"""Prepared pixels of several dates a model reads, one band per channel."""

FeatureMap = Annotated[torch.Tensor, "feature map (batch, channel, y, x)"]
"""Features on a pixel grid, from an encoder level or a decoder."""

Embedding = Annotated[torch.Tensor, "embedding (batch, channel)"]
"""One pooled feature vector per image."""

PrefixTokens = Annotated[torch.Tensor | None, "prefix tokens (batch, token, channel)"]
"""A ViT level's class and register tokens; None for an encoder without them."""

Centroid = Annotated[torch.Tensor, "centroid (batch, coordinate)"]
"""Latitude and longitude of each grid's centre, in degrees, as the `centroid` context reader returns them."""

Year = Annotated[torch.Tensor, "year (batch, time)"]
"""Year of each frame, as the `year` context reader returns it."""

DayOfYear = Annotated[torch.Tensor, "day of year (batch, time)"]
"""Day of year of each frame, from 1, as the `day_of_year` context reader returns it."""

Week = Annotated[torch.Tensor, "ISO week (batch, time)"]
"""ISO week of each frame, as the `week` context reader returns it."""

Hour = Annotated[torch.Tensor, "hour (batch, time)"]
"""Hour of each frame, as the `hour` context reader returns it."""
