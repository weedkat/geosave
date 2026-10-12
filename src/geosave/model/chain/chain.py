"""Compose model modules through their declared named data flow."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from torch import nn

from .routing import ordered_steps, required_inputs
from .step import type_name


class ModelChain(nn.Module):
    """Run one declared method per module in dependency order.

    Args:
        *args: Modules named stage_0, stage_1, and so on.
        **modules: Module instances under explicit stage names.

    Raises:
        ValueError: Positional and explicit stage names collide.
        TypeError: A stage has no step or ambiguous alternatives.
        graphlib.CycleError: Required inputs form a dependency cycle.
    """

    def __init__(self, *args: nn.Module, **modules: nn.Module) -> None:
        super().__init__()
        positional = {f"stage_{index}": module for index, module in enumerate(args)}
        if positional.keys() & modules.keys():
            raise ValueError("Positional stage names collide with named modules")
        modules = positional | modules
        for name, module in modules.items():
            self.add_module(name, module)
        self._steps = ordered_steps(modules)
        self._inputs = required_inputs(self._steps, list(modules))
        self._stage_specs: dict[str, dict[str, object]] | None = None
        # Stages handed over already built; `build_model` narrows this to the
        # ones it did not construct itself.
        self._prebuilt: tuple[str, ...] = tuple(modules)

    @property
    def stage_specs(self) -> dict[str, dict[str, object]]:
        """Return an independent configured-construction recipe.

        Returns:
            Stage selectors and resolved constructor arguments.

        Raises:
            ValueError: A stage was given as a built module.
        """
        if self._stage_specs is None:
            raise ValueError(
                f"stages {list(self._prebuilt)} were given as built modules, which "
                f"carry no construction recipe; give them to `build_model` as "
                f"classes or stage specifications before export"
            )
        return deepcopy(self._stage_specs)

    @property
    def inputs(self) -> dict[str, type]:
        """Return required external names and types in positional order."""
        return dict(self._inputs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Return one native terminal result, several by stage, or named features."""
        if len(args) > len(self._inputs):
            raise TypeError(
                f"ModelChain takes at most {len(self._inputs)} positional arguments"
            )
        positional = dict(zip(self._inputs, args))
        repeated = positional.keys() & kwargs.keys()
        if repeated:
            raise TypeError(
                f"Values supplied both a positional and keyword way: {sorted(repeated)}"
            )
        context = positional | kwargs
        produced: dict[str, object] = {}
        heads: dict[str, object] = {}
        for step in self._steps:
            result = step.method.invoke(step.module, context)
            if step.method.head:
                heads[step.stage] = result
            else:
                values = cast("dict[str, object]", result)
                context.update(values)
                produced.update(values)
        if len(heads) == 1:
            return next(iter(heads.values()))
        return heads if heads else produced

    def __repr__(self) -> str:
        """Return external inputs, execution order, and registered modules."""

        def describe(values: dict[str, type], mark: str = "") -> list[str]:
            return [f"{name}{mark}: {type_name(kind)}" for name, kind in values.items()]

        lines = [f"inputs: {', '.join(describe(self._inputs))}", "", "data flow:"]
        for step in self._steps:
            method = step.method
            arguments = describe(method.inputs) + describe(method.optional_inputs, "?")
            output = (
                type_name(method.result_type)
                if method.head
                else f"{{{', '.join(describe(method.outputs))}}}"
            )
            lines.append(
                f"  {step.stage}: {step.name}({', '.join(arguments)}) -> {output}"
            )
        return "\n".join(lines) + f"\n\n{super().__repr__()}"

    def to_mermaid(self) -> str:
        """Return the data flow as a Mermaid flowchart.

        Each edge carries one value from its producer, or from an external
        input; dashed edges feed optional inputs.

        Returns:
            Mermaid source, rendered by GitHub, JupyterLab, and the Mermaid CLI.

        Examples:
            >>> print(model.to_mermaid())
            flowchart LR
                input_image(["image: Tensor"])
                ...
        """
        producers = {name: step for step in self._steps for name in step.method.outputs}
        consumed = {
            name
            for step in self._steps
            for name in step.method.inputs | step.method.optional_inputs
        }
        optional = {
            name: kind
            for step in self._steps
            for name, kind in step.method.optional_inputs.items()
            if name not in producers and name not in self._inputs
        }
        nodes = [
            f'input_{name}(["{name}: {type_name(kind)}"])'
            for name, kind in (self._inputs | optional).items()
        ]
        nodes += [f'{step.stage}["{step.stage}: {step.name}"]' for step in self._steps]
        nodes += [
            f'result_{step.stage}(["{type_name(step.method.result_type)}"])'
            for step in self._steps
            if step.method.head
        ]

        edges = []
        for step in self._steps:
            method = step.method
            for name, arrow in [
                *((name, "-->") for name in method.inputs),
                *((name, "-.->") for name in method.optional_inputs),
            ]:
                producer = producers.get(name)
                if producer is None:
                    edges.append(f'input_{name} {arrow}|"{name}"| {step.stage}')
                else:
                    kind = type_name(producer.method.outputs[name])
                    edges.append(
                        f'{producer.stage} {arrow}|"{name}: {kind}"| {step.stage}'
                    )
            if method.head:
                edges.append(f"{step.stage} --> result_{step.stage}")
            for name, kind in method.outputs.items():
                if name not in consumed:
                    edges.append(
                        f'{step.stage} -->|"{name}: {type_name(kind)}"| '
                        f'result_{name}(["{name}"])'
                    )
        return "\n".join(["flowchart LR", *(f"    {line}" for line in nodes + edges)])

    def _repr_markdown_(self) -> str:
        """Render the data flow in notebooks that display Mermaid."""
        return f"```mermaid\n{self.to_mermaid()}\n```"
