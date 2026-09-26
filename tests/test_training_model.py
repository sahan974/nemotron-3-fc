import sys
from types import ModuleType

from nemotron3_fc.training.model import load_exact_adapter


class FakeParameter:
    shape = (2, 3)

    def __init__(self):
        self.loaded = None

    def copy_(self, value):
        self.loaded = value


class FakeModel:
    def __init__(self, parameter):
        self.parameter = parameter

    def named_parameters(self):
        return [("base_model.model.layer.lora_A.default.weight", self.parameter)]


class FakeSlice:
    def get_shape(self):
        return [2, 3]


class KeysOnlySafeOpen:
    """Represent the safetensors API, which exposes keys() without iteration."""

    saved_key = "base_model.model.layer.lora_A.weight"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return [self.saved_key]

    def get_slice(self, key):
        assert key == self.saved_key
        return FakeSlice()

    def get_tensor(self, key):
        assert key == self.saved_key
        return "saved tensor"


def test_exact_adapter_loader_uses_safe_open_keys(monkeypatch):
    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda *args, **kwargs: KeysOnlySafeOpen()
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)

    parameter = FakeParameter()
    loaded = load_exact_adapter(FakeModel(parameter), "adapter_model.safetensors")

    assert loaded == 1
    assert parameter.loaded == "saved tensor"
