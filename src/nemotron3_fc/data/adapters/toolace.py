"""ToolACE raw-source adapter."""

from __future__ import annotations

import json
from typing import Any, Mapping

from nemotron3_fc.data.adapters.common import make_name_map, normalize_schema, parse_bracket_calls
from nemotron3_fc.data.schema import Message, ToolCall, canonical_record

TOOL_LIST_MARKER = "Here is a list of functions in JSON format that you can invoke:\n"


class ToolACEAdapter:
    """Convert raw ToolACE conversations to the shared canonical schema."""

    name = "toolace"

    def source_id(self, raw: Mapping[str, Any], index: int) -> str:
        del raw
        return f"toolace:{index}"

    def convert_record(self, raw: Mapping[str, Any], index: int):
        system = raw.get("system")
        if not isinstance(system, str) or TOOL_LIST_MARKER not in system:
            raise ValueError("non_standard_system_prompt")
        try:
            raw_tools, _ = json.JSONDecoder().raw_decode(system.split(TOOL_LIST_MARKER, 1)[1].lstrip())
        except json.JSONDecodeError as error:
            raise ValueError("invalid_standard_tool_list") from error
        if not isinstance(raw_tools, list):
            raise ValueError("standard_tool_list_not_list")

        name_map = make_name_map(raw_tools)
        tools = []
        for tool in raw_tools:
            parameters = normalize_schema(tool.get("parameters", {}))
            if not isinstance(parameters, dict):
                raise ValueError("invalid_tool_parameters")
            tools.append({"type": "function", "function": {"name": name_map[tool["name"]], "description": str(tool.get("description", "")), "parameters": parameters}})

        conversations = raw.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("conversations_not_list")
        messages: list[Message] = []
        pending: list[ToolCall] = []
        for message_index, item in enumerate(conversations):
            if not isinstance(item, Mapping) or not isinstance(item.get("value"), str):
                raise ValueError("invalid_conversation_message")
            role, value = item.get("from"), item["value"]
            if role != "tool" and pending:
                raise ValueError("tool_call_without_following_result")
            if role == "user":
                messages.append(Message(role="user", content=value))
            elif role == "assistant" and value.strip().startswith("["):
                parsed = parse_bracket_calls(value, list(name_map))
                calls = [ToolCall(id=f"call_{index}_{message_index}_{call_index}", name=name_map[call["name"]], arguments=call["arguments"]) for call_index, call in enumerate(parsed)]
                messages.append(Message(role="assistant", tool_calls=tuple(calls)))
                pending = calls
            elif role == "assistant":
                if not value.strip() or _plaintext_is_suspicious(value, list(name_map)):
                    raise ValueError("suspicious_or_empty_assistant_plaintext")
                messages.append(Message(role="assistant", content=value))
            elif role == "tool":
                if not pending:
                    raise ValueError("tool_result_without_previous_call")
                try:
                    results = json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError("invalid_tool_result_json") from error
                if not isinstance(results, list) or len(results) != len(pending):
                    raise ValueError("tool_result_count_mismatch")
                for result, call in zip(results, pending):
                    if not isinstance(result, Mapping) or result.get("name") not in name_map or name_map[result["name"]] != call.name:
                        raise ValueError("tool_result_name_mismatch")
                    content = result.get("results", {key: item for key, item in result.items() if key != "name"})
                    messages.append(Message(role="tool", tool_call_id=call.id, name=call.name, content=content))
                pending = []
            else:
                raise ValueError("unknown_conversation_role")
        return canonical_record(record_id=self.source_id(raw, index), tools=tools, messages=messages, source=self.name, metadata={"source_record_index": index, "original_to_canonical_names": name_map})


def _plaintext_is_suspicious(text: str, names: list[str]) -> bool:
    value = text.lstrip()
    return value.startswith(("{", "<")) or any(token in value[:200] for token in ("-<", "=>", "|[")) or any(value.startswith(name + "(") or value.startswith(name + " (") for name in names)
