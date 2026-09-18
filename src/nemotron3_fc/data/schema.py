"""Canonical tool-calling conversation schema shared by every dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One assistant request to invoke a declared tool."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Tool-call ID must not be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool-call function name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("Tool-call arguments must be an object")
        _ensure_json(self.arguments, "Tool-call arguments")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": dict(self.arguments)}}


@dataclass(frozen=True)
class Message:
    """One canonical conversation message."""

    role: Role
    content: Any = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def validate(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {self.role}")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("Tool messages require tool_call_id")
            if not self.name:
                raise ValueError("Tool messages require the called function name")
            _ensure_json(self.content, "Tool content")
        else:
            if self.tool_call_id is not None or self.name is not None:
                raise ValueError("Only tool messages may contain tool_call_id and name")
            if self.content is not None and not isinstance(self.content, str):
                raise TypeError("Non-tool message content must be a string or null")
        if self.role in {"system", "user"} and not isinstance(self.content, str):
            raise ValueError(f"{self.role.title()} messages require string content")
        if self.role == "assistant" and not self.tool_calls:
            if not isinstance(self.content, str) or not self.content.strip():
                raise ValueError("Assistant text messages require non-empty content")
        for call in self.tool_calls:
            call.validate()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        elif self.role != "tool" or self.content is not None:
            result["content"] = self.content
        if self.role == "tool":
            result.update({"tool_call_id": self.tool_call_id, "name": self.name, "content": self.content})
        return result


@dataclass(frozen=True)
class CanonicalRecord:
    """Dataset-neutral conversation used by training and evaluation."""

    id: str
    tools: tuple[Mapping[str, Any], ...]
    messages: tuple[Message, ...]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported canonical schema version: {self.schema_version}")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Record ID must not be empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Record source must not be empty")
        if not self.messages:
            raise ValueError("Record must contain at least one message")
        if self.messages[0].role != "user":
            raise ValueError("Record must begin with a user message")

        definitions = {_tool_name(tool): _tool_parameters(tool) for tool in self.tools}
        if len(definitions) != len(self.tools):
            raise ValueError("Tool names must be unique within a record")
        calls: dict[str, str] = {}
        results: set[str] = set()
        for message in self.messages:
            message.validate()
            for call in message.tool_calls:
                if call.name not in definitions:
                    raise ValueError(f"Call references undeclared tool: {call.name}")
                if call.id in calls:
                    raise ValueError(f"Duplicate tool-call ID: {call.id}")
                _validate_arguments(call.arguments, definitions[call.name], call.name)
                calls[call.id] = call.name
            if message.role == "tool":
                if message.tool_call_id not in calls:
                    raise ValueError(f"Tool result references unknown call ID: {message.tool_call_id}")
                if message.tool_call_id in results:
                    raise ValueError(f"Duplicate tool result for call ID: {message.tool_call_id}")
                if message.name != calls[message.tool_call_id]:
                    raise ValueError(f"Tool result name does not match call ID: {message.tool_call_id}")
                results.add(message.tool_call_id)
        _ensure_json(self.to_dict(), "Canonical record")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "id": self.id,
            "source": self.source,
            "tools": [dict(tool) for tool in self.tools],
            "messages": [message.to_dict() for message in self.messages],
        }
        result.update(dict(self.metadata))
        return result


def _ensure_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def _tool_name(tool: Mapping[str, Any]) -> str:
    if tool.get("type") != "function":
        raise ValueError("Every tool definition must have type='function'")
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("Tool function definition must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every tool definition requires a non-empty name")
    return name


def _tool_parameters(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool["function"]
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Every tool definition requires an object parameter schema")
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ValueError("Tool parameters require object properties and a required list")
    if any(not isinstance(item, str) for item in required):
        raise ValueError("Required parameter names must be strings")
    return parameters


def _validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any], tool_name: str) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    unknown = set(arguments) - set(properties)
    missing = set(required) - set(arguments)
    if unknown or missing:
        raise ValueError(f"Arguments violate schema for {tool_name}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def message_from_dict(raw: Mapping[str, Any]) -> Message:
    """Build a canonical message from its JSON representation."""
    calls = tuple(
        ToolCall(id=str(call["id"]), name=str(call["function"]["name"]), arguments=call["function"].get("arguments", {}))
        for call in raw.get("tool_calls", ())
    )
    return Message(role=raw["role"], content=raw.get("content"), tool_calls=calls, tool_call_id=raw.get("tool_call_id"), name=raw.get("name"))


def record_from_dict(raw: Mapping[str, Any]) -> CanonicalRecord:
    """Build and validate a canonical record from JSON data."""
    reserved = {"schema_version", "id", "source", "tools", "messages"}
    return canonical_record(
        record_id=str(raw["id"]),
        tools=raw["tools"],
        messages=[message_from_dict(message) for message in raw["messages"]],
        source=str(raw["source"]),
        metadata={key: value for key, value in raw.items() if key not in reserved},
        schema_version=str(raw.get("schema_version", "1.0")),
    )


def canonical_record(*, record_id: str, tools: Sequence[Mapping[str, Any]], messages: Sequence[Message], source: str, metadata: Mapping[str, Any] | None = None, schema_version: str = "1.0") -> CanonicalRecord:
    """Construct and validate an immutable canonical record."""
    record = CanonicalRecord(id=record_id, tools=tuple(tools), messages=tuple(messages), source=source, metadata={} if metadata is None else dict(metadata), schema_version=schema_version)
    record.validate()
    return record
