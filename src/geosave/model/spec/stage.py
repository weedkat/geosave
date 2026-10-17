"""Ordered call declarations for one model processing stage."""

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import Field, RootModel

from .base import Name
from .call import CallSpec


class StageSpec(RootModel[dict[Name, CallSpec]], Mapping[str, CallSpec]):
    """Validate and expose an ordered mapping of named calls.

    Attributes:
        root: Output names mapped to call declarations in execution order.
            Each result is available to later calls under its output name.
            An empty mapping declares no processing.
    """

    root: dict[Name, CallSpec] = Field(default_factory=dict)

    def __getitem__(self, name: str) -> CallSpec:
        """Look up a call declaration without executing it.

        Args:
            name: Declared output name used as the mapping key.

        Returns:
            The stored `CallSpec` for that output.

        Raises:
            KeyError: The output name is not declared.
        """
        return self.root[name]

    def __iter__(self) -> Any:
        """Iterate declared output names in execution order.

        Returns:
            Iterator over mapping keys, without executing their calls.
        """
        return iter(self.root)

    def __len__(self) -> int:
        """Count declared calls without executing them.

        Returns:
            Number of output names in the stage mapping.
        """
        return len(self.root)

    @property
    def external_inputs(self) -> tuple[str, ...]:
        """Return external roots in first-reference order."""
        assigned: set[str] = set()
        external: dict[str, None] = {}
        for output, call in self.root.items():
            for reference in call.references:
                if reference.root not in assigned:
                    external.setdefault(reference.root, None)
            assigned.add(output)
        return tuple(external)

    def validate_inputs(self, names: Collection[str]) -> frozenset[str]:
        """Validate external and ordered references before execution.

        Args:
            names: Runtime root names available before the first call.

        Returns:
            External roots required by the stage, without importing or
            executing any call.

        Raises:
            ValueError: A call reads an unavailable external root or a stage
                result that has not been assigned yet.
        """
        supplied = set(names)
        declared = set(self.root)
        assigned: set[str] = set()
        required: set[str] = set()
        for output, call in self.root.items():
            for name in call.inputs:
                if name in assigned:
                    continue
                if name in supplied:
                    required.add(name)
                elif name in declared:
                    raise ValueError(f"Forward stage reference: {name!r}")
                else:
                    required.add(name)
            assigned.add(output)
        if missing := required - supplied:
            raise ValueError(f"Missing stage inputs: {sorted(missing)}")
        return frozenset(required)

    def run(self, inputs: Mapping[str, Any], /) -> dict[str, Any]:
        """Execute declarations in order without changing the input mapping.

        Args:
            inputs: Available runtime roots by name. Their objects are shared
                with the declared calls, which can mutate them if written to.

        Returns:
            Only declared outputs, in execution order. Each call determines
            its result type, laziness, and side effects.

        Raises:
            ValueError: A required root is missing or a reference points
                forward to a result not yet assigned.
        """
        self.validate_inputs(inputs.keys())
        state = dict(inputs)
        results = {}

        for output, declaration in self.items():
            result = declaration.invoke(declaration.select_inputs(state))
            state[output] = result
            results[output] = result

        return results
