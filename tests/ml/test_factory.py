"""A Registry selects constructors by name or class path and records what it built."""

import sys

import pytest
from torch import nn

from geosave.ml.factory import BuildSpec, Registry


class Encoder(nn.Module):
    def __init__(self, channels: int, depth: int = 2, **options: object) -> None:
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.options = options


def test_a_name_selects_its_constructor_whatever_its_case():
    registry = Registry(nn.Module)
    registry.register("Tiny")(Encoder)

    assert registry.resolve(BuildSpec(name="TINY")) is Encoder
    assert dict(registry) == {"tiny": Encoder}


def test_a_class_path_selects_a_class_of_the_base_type():
    registry = Registry(nn.Module)

    assert registry.resolve(BuildSpec(class_path="torch.nn.Identity")) is nn.Identity
    with pytest.raises(TypeError, match="OrderedDict is not a Module"):
        registry.resolve(BuildSpec(class_path="collections.OrderedDict"))


def test_get_spec_writes_out_defaults_and_keyword_options():
    registry = Registry(nn.Module)
    registry.register("tiny")(Encoder)
    spec = BuildSpec(name="tiny", init_args={"channels": 8, "scale": 0.5})

    full_spec = registry.get_spec(spec)

    assert full_spec.init_args == {"channels": 8, "depth": 2, "scale": 0.5}
    assert spec.init_args == {"channels": 8, "scale": 0.5}
    rebuilt = registry.build(full_spec)
    assert isinstance(rebuilt, Encoder)
    assert (rebuilt.channels, rebuilt.options) == (8, {"scale": 0.5})


def test_build_passes_run_time_values_positionally():
    registry = Registry(object)
    registry.register("pair")(lambda first, second=0: (first, second))

    assert registry.build(BuildSpec(name="pair", init_args={"second": 2}), 1) == (1, 2)


def test_imports_run_on_first_lookup(tmp_path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "registers_entries.py").write_text("")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "registers_entries", raising=False)
    registry = Registry(nn.Module, imports=("registers_entries",))

    assert "registers_entries" not in sys.modules
    len(registry)
    assert "registers_entries" in sys.modules


def test_an_unknown_name_lists_the_known_ones():
    registry = Registry(nn.Module)
    registry.register("tiny")(Encoder)

    with pytest.raises(KeyError, match=r"available: \['tiny'\]"):
        registry.resolve(BuildSpec(name="absent"))


def test_a_class_outside_the_base_type_is_refused():
    registry = Registry(nn.Module)

    with pytest.raises(TypeError, match="dict is not a Module"):
        registry.register("mapping")(dict)


def test_redefining_a_registered_class_replaces_it():
    # A notebook cell run twice defines a new class of the same name and module.
    registry = Registry(nn.Module)

    def define() -> type[nn.Module]:
        class Encoder(nn.Module):
            pass

        return registry.register("tiny")(Encoder)

    define()
    second = define()

    assert registry["tiny"] is second


def test_another_class_under_a_taken_name_is_refused():
    registry = Registry(nn.Module)
    registry.register("tiny")(Encoder)

    with pytest.raises(ValueError, match="already identifies"):
        registry.register("tiny")(nn.Identity)
