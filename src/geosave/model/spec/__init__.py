"""Save model workflows as configuration and execute installed operations.

Glossary:
    Workflow: The complete declaration of how inputs become predictions and how
        those predictions are handled. ModelSpec owns this declaration.
    Operation: An installed Python function or method performing one task.
    Step: A named use of an operation, with configured keyword arguments. The
        YAML field call identifies the operation; kwargs supplies its arguments.
    Input: A declaration of a caller-supplied value or configured default.
        Values remain native Python objects; consuming operations validate them.
    Reference: A !ref name or dotted attribute path connecting an input or step
        result to another step. Attribute access may execute Python properties.
    Entry: The selected step whose return value supplies !ref entry. It may have
        prerequisites of its own and must be used by the selected output path.
    Output: A selected step, or list of steps, whose returned values the caller
        requests. It may be an intermediate result or a prediction-handling step.
    Convergence: Alternative entry paths feeding the same consumer through
        !ref entry. The consuming operation owns validation of its input values.
    Divergence: Several consumers referencing the same earlier step's value.
    Plan: The ordered steps and external inputs needed for the selected entry
        and output. Planning does not import operations or resolve attributes.
    Run: One execution with its own value lookup. Shared dependencies execute
        once; referenced native values retain identity and can be mutated by
        operations. Python functions own batching and resource cleanup.

Saved entry/output defaults can be overridden per run. YAML parsing and reference
serialization live in yaml; native reference traversal lives in reference.
Input and Step are Pydantic models that validate individual declarations.
ModelSpec is a plain workflow class; it owns connections, planning, execution,
and persistence without exposing Pydantic's model construction/copy interface.
Reference.read_value() reads one named value; find_references() discovers
connections; bind_arguments() substitutes native values into keyword arguments.
Runtime values remain Python objects, not Pydantic models. Loading is inert;
running a workflow executes trusted installed Python code.
"""

from .declarations import Input, Step
from .model import ModelSpec, Plan
from .reference import Reference

__all__ = ["Input", "ModelSpec", "Plan", "Reference", "Step"]
