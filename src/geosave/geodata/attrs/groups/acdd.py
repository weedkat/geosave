"""What a whole dataset is, as ACDD discovery attributes on its root."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import model_validator

from .base import AttrsGroup


class ACDD(AttrsGroup):
    """Hold a whole dataset's ACDD keys, the ones catalogs search on.

    It lives on a Dataset's root attrs, where a STAC Collection's fields land
    when a source loads it.

    Attributes:
        id: Identifier of the dataset, a STAC Collection's id.
        title: Short human-readable name.
        summary: Paragraph about the data, a Collection's description.
        keywords: Comma-separated keywords.
        institution: Organization that produced the data.
        license: License of the data, an SPDX identifier where one applies.
        time_coverage_start: ISO 8601 start of the period the data covers.
        time_coverage_end: ISO 8601 end of that period; equal to the start
            for a single instant.

    Raises:
        ValueError: A coverage bound is not an ISO 8601 instant, or the end
            comes before the start.

    Examples:
        >>> ds.gs.acdd.title
        'Sentinel-2 Level-2A'
    """

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    license: str | None = None
    time_coverage_start: str | None = None
    time_coverage_end: str | None = None

    @model_validator(mode="after")
    def _check_time_coverage_is_ordered(self) -> Self:
        """Require ISO 8601 coverage bounds with the end not before the start."""
        bounds = {
            "time_coverage_start": self.time_coverage_start,
            "time_coverage_end": self.time_coverage_end,
        }
        instants = {}
        for key, text in bounds.items():
            if text is None:
                continue
            try:
                instant = np.datetime64(text)
            except ValueError:
                instant = np.datetime64("NaT")
            if np.isnat(instant):
                raise ValueError(
                    f"{key} {text!r} is not an ISO 8601 date or time. Write it "
                    f"like '2024-03-05' or '2024-03-05T06:07:08'."
                )
            instants[key] = instant
        if (
            len(instants) == 2
            and instants["time_coverage_end"] < instants["time_coverage_start"]
        ):
            raise ValueError(
                f"time_coverage_end {self.time_coverage_end!r} is before "
                f"time_coverage_start {self.time_coverage_start!r}. Swap them, "
                f"or give the same value for a single instant."
            )
        return self
