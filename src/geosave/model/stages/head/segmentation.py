from __future__ import annotations

from geosave.model.registry import register_model

from .dense import DenseHead


@register_model("head", "segmentation")
class SegmentationHead(DenseHead):
    """Map one decoded feature map to per-pixel class logits.

    Args:
        num_classes: Number of classes. Class names belong to the label's
            legend, not to the model.
        feature_channels: Channel width of the decoded feature map. Wired from
            the decoder during stage construction.
        input_size: Input spatial size the logits are resized to. Wired from
            the encoder. None leaves the decoder's size.
        hidden_channels: Width of an optional 3x3 conv refinement. None
            projects linearly.
        dropout: Dropout2d probability before the projection.

    Raises:
        ValueError: `num_classes` is below one.

    Examples:
        >>> head = SegmentationHead(num_classes=2, feature_channels=256)
        >>> head.num_classes
        2
    """

    def __init__(
        self,
        num_classes: int,
        feature_channels: int,
        input_size: int | tuple[int, int] | None = None,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_classes < 1:
            raise ValueError(
                f"a segmentation head needs at least one class, got {num_classes}"
            )
        super().__init__(
            num_classes=num_classes,
            feature_channels=feature_channels,
            input_size=input_size,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.num_classes = num_classes
