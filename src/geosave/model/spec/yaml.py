"""Read and write workflow documents with inert reference tags."""

import yaml

from .reference import Reference


class Loader(yaml.SafeLoader):
    """Read recipes without Python object construction or ambiguous YAML keys."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping:
                raise ValueError("Recipe mapping keys must be unique strings")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("Recipe YAML aliases are not supported")
        return super().compose_node(parent, index)


class Dumper(yaml.SafeDumper):
    """Write inert references using the recipe's !ref tag."""


def construct_reference(loader: Loader, node: yaml.Node) -> Reference:
    """Read a scalar reference without evaluating it."""
    if not isinstance(node, yaml.ScalarNode):
        raise ValueError("!ref requires a scalar name or dotted attribute path")
    return Reference(loader.construct_scalar(node))


Loader.add_constructor("!ref", construct_reference)
Dumper.add_representer(
    Reference, lambda dumper, ref: dumper.represent_scalar("!ref", ref.path)
)


def read_yaml(text: str) -> dict[str, object]:
    """Parse a YAML mapping without importing configured operations."""
    document = yaml.load(text, Loader=Loader)
    if not isinstance(document, dict):
        raise ValueError("ModelSpec must be a YAML mapping")
    return document


def write_yaml(document: dict[str, object]) -> str:
    """Serialize a workflow document, preserving reference tags."""
    return yaml.dump(document, Dumper=Dumper, sort_keys=False)
