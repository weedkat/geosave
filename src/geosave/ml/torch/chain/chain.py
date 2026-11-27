"""Compose model modules into a pipe that passes named values."""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self, TypeGuard, get_args, get_origin, get_type_hints

from geosave.ml.torch.typing import ChainModule

from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_model, save_model
from torch import nn

from .registry import BuildSpec


class ModelChain(
    nn.Module,
    ModelHubMixin,
    library_name="geosave",
    repo_url="https://github.com/weedkat/geosave",
    docs_url="https://weedkat.github.io/geosave/",
):
    """Run modules in the order written, each reading named values earlier ones wrote.

    The chain keeps one record of named values. It starts with the chain's
    inputs, the first module's forward parameters. Each module reads its
    forward parameters from the record by name and writes its values under
    the names in its `returns`, adding new names and replacing existing
    ones; see `ChainModule`. A later module can so read any input or any
    earlier module's value, such as the encoder's `centroid` in a head. The
    last module's result is the chain's. Names and type hints are read when
    the chain is built, so a module assigned later must read and return the
    same as the one it replaces.

    Args:
        **modules: Modules by module name, in calling order.

    Raises:
        TypeError: The modules do not connect; see `check_chain`.

    A chain built with `from_config` saves and loads through the Hugging Face
    Hub mixin: `save_pretrained(folder)`, `push_to_hub(repo)` and
    `from_pretrained(folder_or_repo, revision=...)`.

    Examples:
        >>> chain = ModelChain(encoder=SmallCNN(in_channels=4), decoder=..., head=...)
        >>> chain(image=torch.rand(2, 4, 128, 128)).shape
        torch.Size([2, 10, 128, 128])
        >>> chain.save_pretrained("release")
        >>> ModelChain.from_pretrained("release")
    """

    def __init__(self, **modules: nn.Module) -> None:
        super().__init__()

        classes = {}
        for name, module in modules.items():
            classes[name] = type(module)
        self._inputs = check_chain(classes)

        # Names are read once, so forward runs without inspecting and traces under torch.compile.
        self._reads: dict[str, tuple[str, ...]] = {}
        self._returns: dict[str, tuple[str, ...]] = {}
        self._returns_tuple: dict[str, bool] = {}
        for name, module in modules.items():
            cls = type(module)
            parameters = read_forward_inputs(cls)
            self._reads[name] = tuple(parameter.name for parameter in parameters)
            self._returns[name] = tuple(read_returns(cls))
            self._returns_tuple[name] = get_origin(read_return_hint(cls)) is tuple
            self.add_module(name, module)

        self._specs: dict[str, BuildSpec] | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, Any]]) -> Self:
        """Build a chain from a configuration parsed from JSON or YAML.

        The connections are checked from the selected classes' `returns` and
        forward hints before any module is built, so a misconfigured chain
        fails before downloading weights.

        Args:
            config: Module names mapped, in calling order, to a registered
                `name` or a dotted `class_path`, with `init_args` for the
                constructor. Earlier modules supply omitted published
                arguments.

        Returns:
            The configured chain with its construction recipe.

        Raises:
            ValueError: An entry is invalid.
            TypeError: An argument does not fit, or the modules do not connect.

        Examples:
            >>> ModelChain.from_config({
            ...     "encoder": {"name": "dinov3", "init_args": {"in_channels": 4}},
            ...     "decoder": {"name": "unet"},
            ...     "head": {"name": "segmentation", "init_args": {"num_classes": 2}},
            ... })
        """
        specs: dict[str, BuildSpec] = {}
        classes: dict[str, type[nn.Module]] = {}
        for name, entry in config.items():
            specs[name] = BuildSpec.model_validate(entry)
            classes[name] = specs[name].find_class()

        check_chain(classes)

        modules: dict[str, nn.Module] = {}
        for name, spec in specs.items():
            try:
                specs[name] = spec.complete(modules)
                modules[name] = specs[name].build()
            except Exception as error:
                error.add_note(f"While building module {name!r}")
                raise

        chain = cls(**modules)
        chain._specs = specs
        return chain

    @property
    def config(self) -> dict[str, Any]:
        """Return the configuration a release stores.

        Returns:
            Module names mapped to their selector and every constructor
            argument, in calling order, with `pretrained` off.

        Raises:
            ValueError: The chain was built directly from module instances.
        """
        if self._specs is None:
            raise ValueError(
                "A chain built directly from module instances has no construction "
                "recipe. Use ModelChain.from_config to build a chain for export."
            )

        config = {}
        for name, spec in self._specs.items():
            config[name] = spec.release()
        return config

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the configuration as config.json and the weights as model.safetensors."""
        (save_directory / "config.json").write_text(json.dumps(self.config, indent=2))
        save_model(self, str(save_directory / "model.safetensors"))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **modules: Mapping[str, Any],
    ) -> Self:
        """Rebuild the chain from config.json, whose entries arrive as `modules`, and load its weights."""
        chain = cls.from_config(modules)

        weights = Path(model_id) / "model.safetensors"
        if not weights.is_file():
            weights = hf_hub_download(
                model_id,
                "model.safetensors",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        load_model(chain, weights)
        return chain

    @property
    def inputs(self) -> tuple[inspect.Parameter, ...]:
        """Return the chain's inputs: the first module's forward parameters."""
        return self._inputs

    def forward(self, *args: object, **inputs: object) -> object:
        """Run each module on the named values it reads, recording the values it returns.

        Args:
            *args: The chain's inputs in the order `inputs` lists them.
            **inputs: The chain's inputs by name.

        Returns:
            The last module's result, unchanged.
        """
        record: dict[str, Any] = dict(inputs)
        for parameter, value in zip(self._inputs, args):
            record[parameter.name] = value

        result: Any = None
        for name, module in self.named_children():
            arguments = {}
            for parameter in self._reads[name]:
                arguments[parameter] = record[parameter]

            result = module(**arguments)
            values = result if self._returns_tuple[name] else (result,)
            for value_name, value in zip(self._returns[name], values):
                record[value_name] = value

        return result

    def __repr__(self) -> str:
        """Summarize the chain's inputs, data flow and PyTorch module tree."""
        inputs = []
        for parameter in self._inputs:
            inputs.append(f"{parameter.name}: {format_hint(parameter.annotation)}")
        lines = [f"inputs: {', '.join(inputs)}", "", "data flow:"]

        for name, module in self.named_children():
            cls = type(module)
            lines.append(f"  {name}: {cls.__name__}.{format_forward(cls)}")

        return "\n".join(lines) + f"\n\n{super().__repr__()}"

    def to_mermaid(self) -> str:
        """Return the data flow as a Mermaid flowchart.

        Each edge carries one named value from the module that last wrote it,
        or from the chain's inputs, labelled with its name and type.

        Returns:
            Mermaid source, rendered by GitHub, JupyterLab, and the Mermaid CLI.

        Examples:
            >>> print(model.to_mermaid())
            flowchart LR
                input_image(["image: image (batch, band, y, x)"])
                ...
        """
        lines = []
        writers = {}
        for parameter in self._inputs:
            label = f"{parameter.name}: {format_hint(parameter.annotation)}"
            lines.append(f'input_{parameter.name}(["{label}"])')
            writers[parameter.name] = f"input_{parameter.name}"

        last_name = None
        for name, module in self.named_children():
            lines.append(f'{name}["{name}: {type(module).__name__}"]')
            for parameter in read_forward_inputs(type(module)):
                label = f"{parameter.name}: {format_hint(parameter.annotation)}"
                lines.append(f'{writers[parameter.name]} -->|"{label}"| {name}')

            for value_name in self._returns[name]:
                writers[value_name] = name
            last_name = name

        lines.append(f'{last_name} --> result(["result"])')

        return "\n".join(["flowchart LR", *(f"    {line}" for line in lines)])

    def _repr_markdown_(self) -> str:
        """Render the data flow in notebooks that display Mermaid."""
        return f"```mermaid\n{self.to_mermaid()}\n```"


def read_forward_inputs(cls: type[nn.Module]) -> tuple[inspect.Parameter, ...]:
    """Return forward's parameters, `self` left out, each with its resolved type hint.

    Raises:
        TypeError: A parameter has a default. An optional input would let a
            missing value pass silently; a module with another input set is
            another class.
    """
    hints = get_type_hints(cls.forward, include_extras=True)
    parameters = []
    for name, parameter in inspect.signature(cls.forward).parameters.items():
        if name == "self":
            continue
        if parameter.default is not parameter.empty:
            raise TypeError(
                f"{cls.__name__}.forward: {name!r} has a default; inputs are "
                "required, so declare another module for another input set"
            )
        parameters.append(parameter.replace(annotation=hints[name]))
    return tuple(parameters)


def read_return_hint(cls: type[nn.Module]) -> Any:
    """Return forward's return type hint, `Annotated` metadata kept."""
    return get_type_hints(cls.forward, include_extras=True)["return"]


def is_chain_module(cls: type[nn.Module]) -> TypeGuard[type[ChainModule]]:
    """Return whether a module class declares `returns`, as `ChainModule` requires."""
    return hasattr(cls, "returns")


def read_returns(cls: type[nn.Module]) -> dict[str, Any]:
    """Return each name in the class's `returns` mapped to its type hint.

    A `tuple[...]` return hint gives one hint per name, in order; any other
    hint is the one value of a single name.

    Raises:
        TypeError: The class declares no `returns`, or names another number
            of values than forward returns.
    """
    returned = read_return_hint(cls)
    hints = get_args(returned) if get_origin(returned) is tuple else (returned,)

    if not is_chain_module(cls):
        raise TypeError(
            f"{cls.__name__} declares no `returns`, the names of the values its "
            "forward returns"
        )
    if len(hints) != len(cls.returns):
        raise TypeError(
            f"{cls.__name__}.returns names {len(cls.returns)} values; its forward "
            f"returns {len(hints)}"
        )

    returns = {}
    for name, hint in zip(cls.returns, hints):
        returns[name] = hint
    return returns


def check_chain(
    classes: Mapping[str, type[nn.Module]],
) -> tuple[inspect.Parameter, ...]:
    """Walk the record of named values with type hints, before any module is built.

    The record starts with the first module's parameters, the chain's inputs.
    Each module must find every parameter in the record under its name with
    an equal type hint, and then adds or replaces the names in its `returns`.

    Args:
        classes: Module names mapped to module classes, in calling order.

    Returns:
        The chain's inputs, the first module's forward parameters.

    Raises:
        TypeError: A module reads a name no input or earlier module provides,
            reads it with another type hint, or declares its `returns` wrongly.
    """
    inputs: tuple[inspect.Parameter, ...] = ()
    record: dict[str, Any] = {}
    for name, cls in classes.items():
        parameters = read_forward_inputs(cls)
        if not inputs:
            inputs = parameters
            for parameter in parameters:
                record[parameter.name] = parameter.annotation

        for parameter in parameters:
            if parameter.name not in record:
                raise TypeError(
                    f"{name!r} reads {parameter.name}, which no input or "
                    "earlier module provides"
                )
            if record[parameter.name] != parameter.annotation:
                raise TypeError(
                    f"{name!r} reads {parameter.name} as "
                    f"{format_hint(parameter.annotation)}; the record holds "
                    f"{format_hint(record[parameter.name])}"
                )

        record.update(read_returns(cls))
    return inputs


def format_hint(hint: Any) -> str:
    """Format a hint by its description, such as `list[feature map (batch, channel, y, x)]` or `Tensor`."""
    if get_origin(hint) is list:
        return f"list[{format_hint(get_args(hint)[0])}]"
    if get_origin(hint) is typing.Union:
        members = []
        for member in get_args(hint):
            members.append(format_hint(member))
        return " | ".join(members)
    if get_origin(hint) is Annotated:
        return get_args(hint)[1]
    if get_origin(hint):
        return str(hint)
    return getattr(hint, "__name__", str(hint))


def format_forward(cls: type[nn.Module]) -> str:
    """Format forward by names, as `forward(image) -> pyramid, prefix_tokens`."""
    names = []
    for parameter in read_forward_inputs(cls):
        names.append(parameter.name)
    return f"forward({', '.join(names)}) -> {', '.join(read_returns(cls))}"
