import pytest

from nemotron3_fc.data.schema import Message, ToolCall, canonical_record


def test_canonical_record_accepts_declared_call_and_result():
    record = canonical_record(
        record_id="sample-1",
        source="fixture",
        tools=({"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},),
        messages=(
            Message(role="user", content="Weather in Paris?"),
            Message(role="assistant", tool_calls=(ToolCall(id="call-1", name="weather", arguments={"city": "Paris"}),)),
            Message(role="tool", content={"temperature": 20}, tool_call_id="call-1", name="weather"),
            Message(role="assistant", content="It is 20 degrees."),
        ),
    )
    assert record.id == "sample-1"
    assert record.to_dict()["messages"][2]["content"] == {"temperature": 20}


def test_canonical_record_rejects_undeclared_function():
    with pytest.raises(ValueError, match="undeclared tool"):
        canonical_record(
            record_id="sample-2",
            source="fixture",
            tools=(),
            messages=(
                Message(role="user", content="Run it"),
                Message(role="assistant", tool_calls=(ToolCall(id="call-1", name="missing", arguments={}),)),
            ),
        )
