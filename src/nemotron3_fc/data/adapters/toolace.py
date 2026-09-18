"""ToolACE source adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from nemotron3_fc.data.schema import Message, ToolCall, canonical_record


class ToolACEAdapter:
    """Convert ToolACE JSONL records to the shared canonical schema."""

    name = "toolace"

    def load_split(self, path: Path, options: Mapping[str, object] | None = None) -> Iterable:
        del options
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    yield self.convert_record(raw)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid ToolACE record at {path}:{line_number}: {error}") from error

    def convert_record(self, raw: Mapping[str, Any]):
        messages = tuple(self._convert_message(message) for message in raw["messages"])
        metadata = {
            key: raw[key]
            for key in ("schema_version", "source_record_index", "original_to_canonical_names")
            if key in raw
        }
        return canonical_record(
            record_id=str(raw["id"]),
            tools=raw["tools"],
            messages=messages,
            source=str(raw.get("source", self.name)),
            metadata=metadata,
        )

    def _convert_message(self, raw: Mapping[str, Any]) -> Message:
        calls = tuple(
            ToolCall(
                id=str(call["id"]),
                name=str(call["function"]["name"]),
                arguments=call["function"].get("arguments", {}),
            )
            for call in raw.get("tool_calls", ())
        )
        return Message(
            role=raw["role"],
            content=raw.get("content"),
            tool_calls=calls,
            tool_call_id=raw.get("tool_call_id"),
        )

