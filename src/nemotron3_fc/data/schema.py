"""Canonical tool-calling conversation schema shared by every dataset adapter."""

from __future__ import annotations

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
        if not self.id.strip():
            raise ValueError("Tool-call ID must not be empty")
        if not self.name.strip():
            raise ValueError("Tool-call function name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("Tool-call arguments must be an object")


@dataclass(frozen=True)
class Message:
    """One canonical conversation message."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def validate(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {self.role}")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("Only tool messages may contain tool_call_id")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("Message content must be a string or null")
        for call in self.tool_calls:
            call.validate()


@dataclass(frozen=True)
class CanonicalRecord:
    """Dataset-neutral conversation used by training and evaluation."""

    id: str
    tools: tuple[Mapping[str, Any], ...]
    messages: tuple[Message, ...]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("Record ID must not be empty")
        if not self.source.strip():
            raise ValueError("Record source must not be empty")
        if not self.messages:
            raise ValueError("Record must contain at least one message")

        declared_names = {_tool_name(tool) for tool in self.tools}
        outstanding: set[str] = set()
        seen_call_ids: set[str] = set()

        for message in self.messages:
            message.validate()
            for call in message.tool_calls:
                if call.name not in declared_names:
                    raise ValueError(f"Call references undeclared tool: {call.name}")
                if call.id in seen_call_ids:
                    raise ValueError(f"Duplicate tool-call ID: {call.id}")
                seen_call_ids.add(call.id)
                outstanding.add(call.id)
            if message.role == "tool":
                if message.tool_call_id not in outstanding:
                    raise ValueError(f"Tool result references unknown call ID: {message.tool_call_id}")
                outstanding.remove(message.tool_call_id)


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function", tool)
    if not isinstance(function, Mapping):
        raise TypeError("Tool function definition must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every tool definition requires a non-empty name")
    return name


def canonical_record(
    *,
    record_id: str,
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Message],
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalRecord:
    """Construct and validate an immutable canonical record."""
    record = CanonicalRecord(
        id=record_id,
        tools=tuple(tools),
        messages=tuple(messages),
        source=source,
        metadata={} if metadata is None else dict(metadata),
    )
    record.validate()
    return record

