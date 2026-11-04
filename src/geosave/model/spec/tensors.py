"""Validate the tensor boundary without importing the training stack."""

from typing import Annotated, Any, Literal

from pydantic import Field

from .base import Name, SpecModel


class TensorRequirement(SpecModel):
    """Require a sample tensor's shape and dtype.

    Attributes:
        shape: Axis sizes, shared symbolic sizes, or None for an unconstrained axis.
            A batch dimension is supplied separately by the caller.
        dtype: Required PyTorch dtype name, if constrained.
    """

    shape: tuple[Annotated[int, Field(gt=0)] | Name | None, ...]
    dtype: (
        Literal[
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        ]
        | None
    ) = None

    def validate_tensor(
        self, value: Any, *, name: str, sizes: dict[str, int], batched: bool = False
    ) -> None:
        """Check a tensor, sharing symbolic axis sizes with other inputs.

        Args:
            value: Tensor passed to the model.
            name: Model argument named in errors.
            sizes: Symbolic sizes already established by other inputs.
            batched: Whether a leading DataLoader batch axis is present.

        Raises:
            TypeError: The value is not a tensor.
            ValueError: Shape, shared axis sizes, or dtype disagree.
        """
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} requires a torch.Tensor")
        shape = tuple(value.shape)
        if len(shape) != len(self.shape) + int(batched):
            raise ValueError(
                f"{name} shape {shape} does not match {self.shape}; batched={batched}"
            )
        if batched:
            if sizes.setdefault("__batch__", shape[0]) != shape[0]:
                raise ValueError(f"{name} batch size disagrees with other inputs")
            shape = shape[1:]
        for expected, actual in zip(self.shape, shape, strict=True):
            if isinstance(expected, str):
                expected = sizes.setdefault(expected, actual)
            if expected is not None and expected != actual:
                raise ValueError(f"{name} shape {shape} does not match {self.shape}")
        if self.dtype is not None and value.dtype != getattr(torch, self.dtype):
            raise ValueError(f"{name} dtype must be {self.dtype}, got {value.dtype}")
