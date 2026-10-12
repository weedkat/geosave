"""Requirements on native raster structure and its attrs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, JsonValue, model_validator
import xarray as xr
from xarray.core.utils import equivalent

from geosave_engine.geodata import attrs

from .base import SpecModel, Text
from .stac import StacRecipe


class FieldRequirement(SpecModel):
    """Require attr keys, exact values, or one of several accepted values.

    Both sides compare after the text decoding attr groups use, so a
    GeoTIFF's `"0"` meets a requirement of `0`, and `"[0, 1]"` one of `[0, 1]`.

    Args:
        required: Keys that must have non-null values.
        equals: Keys that must equal the supplied values.
        one_of: Keys whose values must belong to the supplied alternatives.
    """

    required: tuple[Text, ...] = ()
    equals: dict[Text, JsonValue] = Field(default_factory=dict)
    one_of: dict[Text, Annotated[tuple[JsonValue, ...], Field(min_length=1)]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_predicates(self) -> Self:
        if len(self.required) != len(set(self.required)):
            raise ValueError("required must not contain duplicate names")
        for key in self.equals.keys() & self.one_of.keys():
            expected = attrs.decode(self.equals[key])
            if not any(
                equivalent(expected, attrs.decode(value)) for value in self.one_of[key]
            ):
                raise ValueError(f"Conflicting equals and one_of for {key!r}")
        return self

    @property
    def keys(self) -> set[str]:
        """Return every key constrained by these requirements."""
        return set(self.required) | self.equals.keys() | self.one_of.keys()

    def validate_attrs(self, found: Mapping[str, object], *, where: str) -> None:
        """Check one attrs mapping without reading pixels.

        Args:
            found: Attrs of the Dataset, one variable, or one coordinate.
            where: Scope named in errors, such as `data_vars.red`.

        Raises:
            ValueError: A constrained key is absent or null, or its value differs.
        """
        for key in sorted(self.keys):
            value = attrs.decode(found.get(key))
            if value is None:
                raise ValueError(f"{where}.{key} is required")
            if key in self.equals and not equivalent(
                value, attrs.decode(self.equals[key])
            ):
                raise ValueError(
                    f"{where}.{key} must equal {self.equals[key]!r}, got {value!r}"
                )
            if key in self.one_of and not any(
                equivalent(value, attrs.decode(choice)) for choice in self.one_of[key]
            ):
                raise ValueError(
                    f"{where}.{key} must be one of {self.one_of[key]!r}, got {value!r}"
                )


class AttrsRequirement(SpecModel):
    """Requirements on the root, data-variable, and coordinate attrs.

    Args:
        root: Attrs required on the Dataset itself.
        data_vars: Per-variable requirements; `*` applies to selected variables.
        coords: Per-coordinate requirements; `*` applies to all coordinates.
    """

    root: FieldRequirement = Field(default_factory=FieldRequirement)
    data_vars: dict[Text, FieldRequirement] = Field(default_factory=dict)
    coords: dict[Text, FieldRequirement] = Field(default_factory=dict)


class RasterRequirement(SpecModel):
    """Select and validate a raster using native variables or positional channels.

    Args:
        type: Requirement kind, always raster.
        variables: Required data variables in their selected order.
        channels: First N channels; mutually exclusive with variables.
        coordinates: Coordinate arrays that must be present.
        stac: Optional recipe for acquiring the raster from STAC.
        dims: Exact dimension order required for each selected variable, if set.
        dtypes: Accepted stored dtypes, if constrained.
        require_crs: Require a locatable geospatial grid.
        resolution: Positive pixel size, scalar or (x, y), in CRS units.
        attrs: Requirements on the raster's attrs, by key.
    """

    type: Literal["raster"] = "raster"
    variables: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    channels: Annotated[int, Field(gt=0)] | None = None
    coordinates: tuple[Text, ...] = ()
    stac: StacRecipe | None = None
    dims: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    dtypes: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    require_crs: bool = False
    resolution: (
        Annotated[float, Field(gt=0)]
        | tuple[Annotated[float, Field(gt=0)], Annotated[float, Field(gt=0)]]
        | None
    ) = None
    attrs: AttrsRequirement = Field(default_factory=AttrsRequirement)

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if (self.variables is None) == (self.channels is None):
            raise ValueError("Exactly one of variables or channels is required")
        if self.variables is not None and len(self.variables) != len(
            set(self.variables)
        ):
            raise ValueError("variables must not contain duplicate names")
        if self.dims is not None and len(self.dims) != len(set(self.dims)):
            raise ValueError("dims must not contain duplicate names")
        if len(self.coordinates) != len(set(self.coordinates)):
            raise ValueError("coordinates must not contain duplicate names")
        if self.dtypes is not None:
            for dtype in self.dtypes:
                try:
                    np.dtype(dtype)
                except TypeError as error:
                    raise ValueError(f"Unknown raster dtype {dtype!r}") from error
        unknown = self.attrs.data_vars.keys() - set(self.variables or ()) - {"*"}
        if unknown:
            raise ValueError(f"Metadata names unselected variables: {sorted(unknown)}")
        if (
            self.variables is not None
            and self.stac is not None
            and self.stac.load.bands is not None
            and tuple(self.stac.load.bands) != self.variables
        ):
            raise ValueError("STAC load bands must match raster variables in order")
        return self

    def select_raster(self, raster: xr.Dataset) -> xr.Dataset:
        """Return a selected lazy view after checking structure and attrs.

        Args:
            raster: Dataset, possibly backed by Dask.

        Returns:
            Dataset holding the declared variables in order, or the first
            `channels` data variables.

        Raises:
            TypeError: The raster is not a Dataset.
            KeyError: A declared variable, coordinate, or attrs member is absent.
            ValueError: The raster has too few channels, or its grid, dims,
                dtypes, or attrs miss a requirement.

        Examples:
            >>> RasterRequirement(variables=("B08", "B04")).select_raster(scene)
            <xarray.Dataset> ...
        """
        if not isinstance(raster, xr.Dataset):
            raise TypeError(
                f"a raster requirement takes a Dataset, got {type(raster).__name__}"
            )
        if self.variables is not None:
            if missing := [n for n in self.variables if n not in raster.data_vars]:
                raise KeyError(f"Missing raster variables: {missing}")
            selected = raster[list(self.variables)]
        else:
            assert self.channels is not None
            names = list(raster.data_vars)
            if len(names) < self.channels:
                raise ValueError(
                    f"Raster requires {self.channels} channels, got {len(names)}"
                )
            selected = raster[names[: self.channels]]

        if missing := [n for n in self.coordinates if n not in selected.coords]:
            raise KeyError(f"Missing raster coordinates: {missing}")
        self._validate_grid(selected)
        dtypes = None if self.dtypes is None else [np.dtype(d) for d in self.dtypes]
        for name, variable in selected.data_vars.items():
            if self.dims is not None and variable.dims != self.dims:
                raise ValueError(
                    f"{name} requires dims {self.dims}, got {variable.dims}"
                )
            if dtypes is not None and variable.dtype not in dtypes:
                raise ValueError(
                    f"{name} requires dtype in {self.dtypes}, got {variable.dtype}"
                )
        self._validate_attrs(selected)
        return selected

    def _validate_grid(self, raster: xr.Dataset) -> None:
        """Check the CRS and pixel size the requirement asks for."""
        if self.require_crs and not raster.gs.is_georeferenced:
            raise ValueError("Raster requires a locatable CRS and grid")
        if self.resolution is None:
            return
        resolution = raster.gs.resolution
        if resolution is None:
            raise ValueError("Raster resolution requires a locatable CRS and grid")
        expected = (
            (self.resolution, self.resolution)
            if isinstance(self.resolution, float)
            else self.resolution
        )
        actual = (abs(resolution.x), abs(resolution.y))
        if not np.allclose(actual, expected, rtol=1e-6, atol=0):
            raise ValueError(f"Raster requires resolution {expected}, got {actual}")

    def _validate_attrs(self, raster: xr.Dataset) -> None:
        """Check the attrs of the Dataset, its variables and its coordinates."""
        self.attrs.root.validate_attrs(raster.attrs, where="root")
        for scope, declared, members in (
            ("data_vars", self.attrs.data_vars, raster.data_vars),
            ("coords", self.attrs.coords, raster.coords),
        ):
            for name, requirement in declared.items():
                for target in members if name == "*" else (name,):
                    if target not in members:
                        raise KeyError(f"Missing {scope}.{target}")
                    requirement.validate_attrs(
                        members[target].attrs, where=f"{scope}.{target}"
                    )
