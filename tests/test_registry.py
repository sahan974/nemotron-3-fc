import pytest

from nemotron3_fc.data.adapters.toolace import ToolACEAdapter
from nemotron3_fc.data.adapters.xlam import XLAMAdapter
from nemotron3_fc.data.registry import DatasetRegistry


def test_registry_resolves_registered_adapter():
    registry = DatasetRegistry()
    registry.register("toolace", ToolACEAdapter)
    assert isinstance(registry.create("TOOLACE"), ToolACEAdapter)


def test_registry_reports_available_adapters():
    registry = DatasetRegistry()
    registry.register("toolace", ToolACEAdapter)
    with pytest.raises(KeyError, match="Available: toolace"):
        registry.create("custom")


def test_registry_can_hold_multiple_source_adapters():
    registry = DatasetRegistry()
    registry.register("toolace", ToolACEAdapter)
    registry.register("xlam", XLAMAdapter)
    assert registry.names() == ("toolace", "xlam")
