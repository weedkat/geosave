"""Configure and execute model workflows on native Python values."""

from copy import deepcopy
from graphlib import CycleError, TopologicalSorter
from importlib import import_module
import keyword
from typing import ClassVar, TypedDict, overload

from pydantic import TypeAdapter

from geosave.geodata.io.storage import Location, StorageOptions, location

from .declarations import Input, Step
from .reference import Reference, bind_arguments, find_references
from .yaml import read_yaml, write_yaml


_DOCUMENT_FIELDS = {"schema_version", "inputs", "steps", "entry", "output"}


class Plan(TypedDict):
    """Selected entry/output, ordered steps, and required external inputs."""

    entry: str
    output: str | list[str]
    steps: tuple[str, ...]
    inputs: tuple[str, ...]


class ModelSpec:
    """Save and execute a workflow of installed Python operations.

    Loading and planning do not import configured operations. Running a workflow
    executes trusted Python code, including attribute properties. Operations own
    batching, branching, resource cleanup, and validation of the values they use.

    Args:
        steps: Validated Step declarations with literal or referenced arguments.
        inputs: Validated Input declarations, optionally with default values.
        entry: Default step supplying !ref entry. May have its own prerequisites.
        output: Default step or list of steps whose values are returned.

    Examples:
        >>> spec = ModelSpec.load("model_spec.yaml")
        >>> prediction = spec.run(raster=raster)
        >>> path = spec.run(output="saved", raster=raster, destination="map.tif")
    """

    filename: ClassVar[str] = "model_spec.yaml"

    def __init__(
        self,
        *,
        steps: dict[str, Step],
        inputs: dict[str, Input] | None = None,
        entry: str | None = None,
        output: str | list[str] | None = None,
    ) -> None:
        self.inputs = dict(inputs) if inputs is not None else {}
        self.steps = dict(steps)
        self.entry = entry
        self.output = list(output) if isinstance(output, list) else output
        self._check_connections()

    def _check_connections(self) -> None:
        """Check workflow names, references, and saved entry/output selections."""
        if self.inputs.keys() & {"entry", "output"}:
            raise ValueError("entry and output are reserved input names")
        if "entry" in self.steps:
            raise ValueError("entry is reserved for the selected entry step's value")
        if self.inputs.keys() & self.steps.keys():
            raise ValueError("Input and step names must be distinct")

        for name in self.inputs:
            self._check_name(name)

        known = self.inputs.keys() | self.steps.keys() | {"entry"}
        for name, step in self.steps.items():
            self._check_name(name)
            for ref in step.find_references():
                if ref.root not in known:
                    raise ValueError(
                        f"Step {name!r} has unknown reference {ref.path!r}. "
                        "Declare that input or step, or correct the reference."
                    )

        if self.entry is not None:
            self._select_entry(self.entry)
        if self.output is not None:
            self._select_output(self.output)

    @staticmethod
    def _check_name(name: str) -> None:
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            raise ValueError(
                f"Input and step names must be identifiers without dots: {name!r}"
            )

    @classmethod
    def from_dict(cls, document: dict[str, object]) -> "ModelSpec":
        """Construct a workflow from its saved schema version 1 declarations."""
        if not isinstance(document, dict):
            raise ValueError("ModelSpec must be a mapping")
        version = document.get("schema_version")
        if type(version) is not int or version != 1:
            raise ValueError("ModelSpec requires schema_version: 1")
        unknown = document.keys() - _DOCUMENT_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown workflow fields: {sorted(unknown)}. "
                "Declare named operations inside the 'steps' mapping."
            )

        try:
            inputs = TypeAdapter(dict[str, Input]).validate_python(
                document.get("inputs", {}), strict=True
            )
        except ValueError as error:
            error.add_note(
                "Cannot load workflow inputs. Check the named input declaration above."
            )
            raise
        try:
            steps = TypeAdapter(dict[str, Step]).validate_python(
                document.get("steps", {}), strict=True
            )
        except ValueError as error:
            error.add_note(
                "Cannot load workflow steps. Check the named step declaration above."
            )
            raise

        entry = TypeAdapter(str | None).validate_python(
            document.get("entry"), strict=True
        )
        output = TypeAdapter(str | list[str] | None).validate_python(
            document.get("output"), strict=True
        )
        return cls(inputs=inputs, steps=steps, entry=entry, output=output)

    @classmethod
    def from_yaml(cls, text: str) -> "ModelSpec":
        """Parse a YAML workflow without importing configured operations."""
        return cls.from_dict(read_yaml(text))

    @classmethod
    def load(
        cls, path: Location, *, storage_options: StorageOptions | None = None
    ) -> "ModelSpec":
        """Read a workflow from local or remote storage without executing it."""
        text = location(path, storage_options).read_text(encoding="utf-8")
        return cls.from_yaml(text)

    def save(
        self, path: Location, *, storage_options: StorageOptions | None = None
    ) -> str:
        """Write declarations and selection defaults, preserving !ref tags."""
        document: dict[str, object] = {"schema_version": 1}
        if self.entry is not None:
            document["entry"] = self.entry
        if self.output is not None:
            document["output"] = self.output
        inputs = {}
        for name, declaration in self.inputs.items():
            # Keep omitted defaults distinct from explicit nulls, and preserve
            # native values for YAML rather than recursively serializing them.
            fields = dict(declaration)
            inputs[name] = {
                key: value
                for key, value in fields.items()
                if key in declaration.model_fields_set
            }
        document["inputs"] = inputs
        document["steps"] = {name: dict(step) for name, step in self.steps.items()}

        target = location(path, storage_options)
        target.write_text(write_yaml(document), encoding="utf-8")
        return str(target)

    def _select_entry(self, entry: str | None) -> str:
        if entry is None:
            raise ValueError(
                "No workflow entry was selected. Set 'entry' in the configuration "
                "or pass entry='step_name' when calling run() or plan()."
            )
        if not isinstance(entry, str) or entry not in self.steps:
            raise ValueError(
                f"Unknown entry {entry!r}. Choose a declared step: {list(self.steps)}"
            )
        return entry

    def _select_output(self, output: str | list[str] | None) -> str | list[str]:
        if output is None:
            raise ValueError(
                "No workflow output was selected. Set 'output' in the configuration "
                "or pass output='step_name' when calling run() or plan()."
            )
        targets = [output] if isinstance(output, str) else output
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                "Select one output step or a nonempty list of output steps"
            )
        for name in targets:
            if not isinstance(name, str) or name not in self.steps:
                raise ValueError(
                    f"Unknown output {name!r}. Choose a declared step: {list(self.steps)}"
                )
        return list(output) if isinstance(output, list) else output

    def plan(
        self, *, entry: str | None = None, output: str | list[str] | None = None
    ) -> Plan:
        """Order the selected output's dependencies without importing operations.

        Omitted selections use saved defaults. The entry supplies !ref entry and
        must participate in the selected dependencies. Its prerequisites also run.
        Only the selected dependencies are checked for cycles.
        """
        entry = self._select_entry(self.entry if entry is None else entry)
        output = self._select_output(self.output if output is None else output)
        targets = [output] if isinstance(output, str) else output

        dependencies = {}
        needed_inputs = set()
        pending = list(reversed(targets))
        while pending:
            name = pending.pop()
            if name in dependencies:
                continue
            needed = []
            for ref in self.steps[name].find_references():
                root = ref.root
                if root in self.inputs:
                    needed_inputs.add(root)
                    continue
                if root == "entry":
                    root = entry
                if root not in needed:
                    needed.append(root)
            dependencies[name] = needed
            pending.extend(reversed(needed))

        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            error.add_note(
                f"Cannot produce output {output!r} from entry {entry!r}: "
                "the selected steps depend on each other in a cycle. "
                "Correct their references so dependencies flow forward."
            )
            raise
        if entry not in dependencies:
            raise ValueError(
                f"Output {output!r} does not use entry {entry!r}. "
                "Choose an entry on this path, or connect the shared flow with !ref entry."
            )
        inputs = tuple(name for name in self.inputs if name in needed_inputs)
        return {"entry": entry, "output": output, "steps": order, "inputs": inputs}

    def _bind_inputs(
        self, plan: Plan, arguments: dict[str, object]
    ) -> dict[str, object]:
        unused = arguments.keys() - set(plan["inputs"])
        if unused:
            raise ValueError(
                f"Inputs unused by output {plan['output']!r}: {sorted(unused)}. "
                "Remove them or select an output that uses them."
            )

        values: dict[str, object] = {}
        for name in plan["inputs"]:
            definition = self.inputs[name]
            if name in arguments:
                value = arguments[name]
            elif "default" in definition.model_fields_set:
                value = deepcopy(definition.default)
            else:
                uses = []
                for step_name in plan["steps"]:
                    step = self.steps[step_name]
                    if any(ref.root == name for ref in find_references(step.call)):
                        uses.append(f"step {step_name!r}'s operation")
                    for argument, configured in step.kwargs.items():
                        if any(ref.root == name for ref in find_references(configured)):
                            uses.append(f"step {step_name!r}, argument {argument!r}")
                raise ValueError(
                    f"Cannot produce output {plan['output']!r}: input {name!r} is missing.\n"
                    f"Required by: {'; '.join(uses)}.\n"
                    f"Pass {name}=... to spec.run(), alongside the other required inputs."
                )

            values[name] = value
        return values

    @overload
    def run(
        self,
        /,
        *,
        output: list[str],
        entry: str | None = None,
        **arguments: object,
    ) -> dict[str, object]: ...

    @overload
    def run(
        self,
        /,
        *,
        output: str | None = None,
        entry: str | None = None,
        **arguments: object,
    ) -> object: ...

    def run(
        self,
        /,
        *,
        entry: str | None = None,
        output: str | list[str] | None = None,
        **arguments: object,
    ) -> object:
        """Execute selected steps once and return native output value(s).

        A list of outputs returns a dictionary keyed by step name. Shared steps
        run once. Missing or unused inputs fail before operation imports. Input
        defaults are copied; referenced native values are passed unchanged.
        Operations are loaded and called in execution order. Python validates
        their call arguments; earlier steps may have executed before a failure.

        Returns:
            The selected operation's native return value. Its concrete type is
            defined by that operation, not by ModelSpec. An explicit output list
            returns a dictionary from step names to their native return values.
        """
        plan = self.plan(entry=entry, output=output)
        values = self._bind_inputs(plan, arguments)

        for name in plan["steps"]:
            step = self.steps[name]
            target = step.call
            try:
                if isinstance(target, Reference):
                    function = target.read_value(values)
                else:
                    module, attribute = target.rsplit(".", 1)
                    try:
                        function = getattr(import_module(module), attribute)
                    except Exception as error:
                        error.add_note(
                            f"Cannot load operation {target!r}. "
                            "Check that its package is installed and its module "
                            "and attribute names are correct."
                        )
                        raise

                if not callable(function):
                    raise TypeError(
                        f"Configured operation is a {type(function).__name__}, "
                        "not a callable. Configure a function or method."
                    )
                kwargs = bind_arguments(step.kwargs, values)
                values[name] = function(**kwargs)
                if name == plan["entry"]:
                    values["entry"] = values[name]
            except Exception as error:
                operation = target.path if isinstance(target, Reference) else target
                error.add_note(
                    f"Step {name!r} failed while producing output {plan['output']!r}.\n"
                    f"Configured operation: {operation}."
                )
                raise

        selected = plan["output"]
        if isinstance(selected, str):
            return values[selected]
        return {name: values[name] for name in selected}
