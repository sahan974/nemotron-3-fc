"""xLAM 60K raw-source adapter."""

from __future__ import annotations

import json
from typing import Any, Mapping

from nemotron3_fc.data.adapters.common import make_name_map, split_outside_literals
from nemotron3_fc.data.schema import Message, ToolCall, canonical_record


class XLAMAdapter:
    """Convert xLAM function-calling records to the shared canonical schema."""

    name = "xlam"

    def source_id(self, raw: Mapping[str, Any], index: int) -> str:
        del index
        return f"xlam:{raw.get('id')}"

    def convert_record(self, raw: Mapping[str, Any], index: int):
        del index
        source_id, query = raw["id"], raw["query"]
        raw_tools = _decode_json_field(raw["tools"], "tools")
        answers = _decode_json_field(raw["answers"], "answers")
        if not isinstance(query, str) or not isinstance(raw_tools, list) or not isinstance(answers, list) or not answers:
            raise ValueError("invalid_query_tools_or_answers")

        unique, seen, removed = [], {}, 0
        for tool in raw_tools:
            if not isinstance(tool, Mapping) or not isinstance(tool.get("name"), str) or not tool["name"]:
                raise ValueError("invalid_tool_name")
            if tool["name"] in seen:
                if tool != seen[tool["name"]]:
                    raise ValueError("conflicting_duplicate_tool_name")
                removed += 1
            else:
                seen[tool["name"]] = tool
                unique.append(tool)

        name_map = make_name_map(unique)
        tools = []
        for tool in unique:
            parameters = tool.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError("invalid_tool_parameters")
            properties = {}
            for name, specification in parameters.items():
                if not isinstance(specification, Mapping):
                    raise ValueError("invalid_parameter_specification")
                property_schema = xlam_type_schema(specification.get("type"))
                if "description" in specification:
                    property_schema["description"] = str(specification["description"])
                if "default" in specification:
                    property_schema["default"] = specification["default"]
                properties[name] = property_schema
            tools.append({"type": "function", "function": {"name": name_map[tool["name"]], "description": str(tool.get("description", "")), "parameters": {"type": "object", "properties": properties, "required": []}}})

        calls = []
        for call_index, answer in enumerate(answers):
            if not isinstance(answer, Mapping) or answer.get("name") not in name_map or not isinstance(answer.get("arguments"), Mapping):
                raise ValueError("answer_not_declared_function_call")
            calls.append(ToolCall(id=f"call_{source_id}_1_{call_index}", name=name_map[answer["name"]], arguments=answer["arguments"]))
        return canonical_record(record_id=f"xlam:{source_id}", tools=tools, messages=(Message(role="user", content=query), Message(role="assistant", tool_calls=tuple(calls))), source=self.name, metadata={"source_record_index": source_id, "original_to_canonical_names": name_map, "source_duplicate_tool_definitions_removed": removed})


def _decode_json_field(value: Any, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid_{label}_json") from error
    return value


def xlam_type_schema(label: Any) -> dict[str, Any]:
    """Translate xLAM's compact Python-like type labels to JSON Schema."""
    base = str(label).split(", optional")[0].split(", default")[0].strip()
    simple = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
    if base in simple:
        return {"type": simple[base]}
    if base.startswith("List[") and base.endswith("]"):
        return {"type": "array", "items": xlam_type_schema(base[5:-1])}
    if base.startswith("Union[") and base.endswith("]"):
        return {"anyOf": [xlam_type_schema(part) for part in split_outside_literals(base[6:-1], ",")]}
    if base.startswith("Tuple[") and base.endswith("]"):
        parts = split_outside_literals(base[6:-1], ",")
        return {"type": "array", "prefixItems": [xlam_type_schema(part) for part in parts], "minItems": len(parts), "maxItems": len(parts)}
    if base in ("list", "List", "set"):
        return {"type": "array"}
    if base == "Dict":
        return {"type": "object"}
    return {}
