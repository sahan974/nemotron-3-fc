import json

from nemotron3_fc.data.adapters.toolace import TOOL_LIST_MARKER, ToolACEAdapter
from nemotron3_fc.data.adapters.xlam import XLAMAdapter


def test_toolace_adapter_preserves_multiturn_calls_and_results():
    tools = [{"name": "weather.lookup", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "str"}}, "required": ["city"]}}]
    raw = {
        "system": TOOL_LIST_MARKER + json.dumps(tools),
        "conversations": [
            {"from": "user", "value": "Weather in Paris?"},
            {"from": "assistant", "value": "[weather.lookup(city='Paris')]"},
            {"from": "tool", "value": json.dumps([{"name": "weather.lookup", "results": {"temperature": 20}}])},
            {"from": "assistant", "value": "It is 20 degrees."},
        ],
    }
    record = ToolACEAdapter().convert_record(raw, 7)
    assert record.id == "toolace:7"
    assert record.tools[0]["function"]["name"] == "weather_lookup"
    assert record.messages[1].tool_calls[0].arguments == {"city": "Paris"}
    assert record.messages[2].content == {"temperature": 20}
    assert sum(message.role == "user" for message in record.messages) == 1


def test_xlam_adapter_removes_identical_duplicate_definitions():
    tool = {"name": "lookup", "description": "Lookup", "parameters": {"query": {"type": "str", "description": "Term"}}}
    raw = {
        "id": "x-1",
        "query": "Find alpha",
        "tools": json.dumps([tool, tool]),
        "answers": json.dumps([{"name": "lookup", "arguments": {"query": "alpha"}}]),
    }
    record = XLAMAdapter().convert_record(raw, 0)
    assert len(record.tools) == 1
    assert record.metadata["source_duplicate_tool_definitions_removed"] == 1
    assert record.messages[1].tool_calls[0].name == "lookup"
