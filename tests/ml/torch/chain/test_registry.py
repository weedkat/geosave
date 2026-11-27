"""A Registry selects constructors by name; a BuildSpec by name or class path."""

import sys

import pytest
import torch
from torch import nn

from geosave.ml.torch.chain.published import Published
from geosave.ml.torch.chain.registry import MODULES, BuildSpec, Registry


class Encoder(nn.Module):
    channels: Published[int]

    def __init__(self, channels: int, depth: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.depth = depth

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pyramid = image
        return pyramid


class Decoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, pyramid: torch.Tensor) -> torch.Tensor:
        return pyramid


def test_a_name_selects_its_class_whatever_its_case():
    registry = Registry()
    registry.register("Tiny")(Encoder)

    assert registry["TINY"] is Encoder
    assert dict(registry) == {"tiny": Encoder}


def test_a_spec_selects_a_registered_or_an_importable_class():
    assert BuildSpec(name="dinov3").find_class() is MODULES["dinov3"]
    assert BuildSpec(class_path="torch.nn.Identity").find_class() is nn.Identity


def test_a_completed_spec_writes_out_published_and_default_arguments():
    spec = BuildSpec(class_path=f"{__name__}.Decoder").complete(
        {"encoder": Encoder(channels=8)}
    )

    assert spec.init_args == {"channels": 8}
    assert isinstance(spec.build(), Decoder)


def test_a_release_turns_upstream_weights_off():
    spec = BuildSpec(class_path=f"{__name__}.Encoder", init_args={"channels": 8})

    released = spec.complete({}).release()

    assert released == {
        "class_path": f"{__name__}.Encoder",
        "init_args": {"channels": 8, "depth": 2, "pretrained": False},
    }
    assert spec.init_args == {"channels": 8}


def test_imports_run_on_first_lookup(tmp_path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "registers_entries.py").write_text("")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "registers_entries", raising=False)
    registry = Registry(imports=("registers_entries",))

    assert "registers_entries" not in sys.modules
    len(registry)
    assert "registers_entries" in sys.modules


def test_an_unknown_name_lists_the_known_ones():
    registry = Registry()
    registry.register("tiny")(Encoder)

    with pytest.raises(KeyError, match=r"available: \['tiny'\]"):
        registry["absent"]


def test_redefining_a_registered_class_replaces_it():
    # A notebook cell run twice defines a new class of the same name and module.
    registry = Registry()

    def define() -> type[nn.Module]:
        class Encoder(nn.Module):
            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return image

        return registry.register("tiny")(Encoder)

    define()
    second = define()

    assert registry["tiny"] is second


def test_another_class_under_a_taken_name_is_refused():
    registry = Registry()
    registry.register("tiny")(Encoder)

    with pytest.raises(TypeError, match="'tiny' is already registered to Encoder"):
        registry.register("tiny")(Decoder)


def test_a_class_belongs_to_the_family_of_the_package_defining_it():
    assert MODULES.families == ("encoder", "decoder", "head", "monolith", "custom")
    assert MODULES.find_family(MODULES["clay"]) == "encoder"
    assert MODULES.find_family(MODULES["ibm_granite_biomass"]) == "monolith"
    assert MODULES.find_family(Encoder) == "custom"


def test_library_names_are_registered():
    assert {"dinov3", "unet", "dpt", "segmentation", "ibm_granite_biomass"} <= set(
        MODULES
    )
