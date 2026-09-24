"""Shared parsing helpers for built-in dataset adapters."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from typing import Any


def split_outside_literals(text: str, separator: str) -> list[str]:
    """Split on a character while respecting nested containers and quoted strings."""
    parts: list[str] = []
    start, stack, quote, escaped = 0, [], None, False
    pairs = {")": "(", "]": "[", "}": "{"}
    # Separators inside strings or nested containers are argument data rather
    # than boundaries between calls.
    for position, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in ("'", '"'):
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}":
            if not stack or stack.pop() != pairs[character]:
                raise ValueError("Unbalanced brackets")
        elif character == separator and not stack:
            parts.append(text[start:position].strip())
            start = position + 1
    if quote is not None or stack:
        raise ValueError("Unclosed quote or bracket")
    parts.append(text[start:].strip())
    return parts


def decode_arguments(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    arguments: dict[str, Any] = {}
    for item in split_outside_literals(text, ","):
        pair = split_outside_literals(item, "=")
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise ValueError("Invalid argument assignment")
        name = ast.literal_eval(pair[0]) if pair[0][0] in ("'", '"') else pair[0]
        if not isinstance(name, str) or not name or name in arguments:
            raise ValueError("Invalid or duplicate argument name")
        # literal_eval supports source values such as tuples and quoted strings
        # without executing arbitrary code; JSON is the strict fallback.
        try:
            value = ast.literal_eval(pair[1])
        except (ValueError, SyntaxError):
            value = json.loads(pair[1])
        json.dumps(value, allow_nan=False)
        arguments[name] = value
    return arguments


def parse_bracket_calls(value: str, declared_names: Sequence[str]) -> list[dict[str, Any]]:
    """Parse ToolACE's bracketed function-call representation."""
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Tool call is not bracketed")
    inner = text[1:-1].strip()
    if not inner:
        raise ValueError("Empty bracketed call")
    names = sorted(declared_names, key=len, reverse=True)
    calls: list[dict[str, Any]] = []
    position = 0
    while position < len(inner):
        name = next(
            (
                candidate
                for candidate in names
                if inner.startswith(candidate, position) and inner[position + len(candidate) :].lstrip().startswith("(")
            ),
            None,
        )
        if name is None:
            raise ValueError("Undeclared tool or alternate call syntax")
        position += len(name)
        while position < len(inner) and inner[position].isspace():
            position += 1
        position += 1
        argument_start, depth, quote, escaped = position, 1, None, False
        while position < len(inner) and depth:
            character = inner[position]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in ("'", '"'):
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            position += 1
        if depth or quote is not None:
            raise ValueError("Unclosed call arguments")
        calls.append({"name": name, "arguments": decode_arguments(inner[argument_start : position - 1])})
        while position < len(inner) and inner[position].isspace():
            position += 1
        if position == len(inner):
            break
        if inner[position] != ",":
            raise ValueError("Unexpected call separator")
        position += 1
        while position < len(inner) and inner[position].isspace():
            position += 1
        if position == len(inner):
            raise ValueError("Trailing call separator")
    return calls


def make_name_map(tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Produce unique ASCII tool names accepted by model runtimes."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    # Runtime-safe aliases preserve a one-to-one mapping to original names.
    for index, tool in enumerate(tools):
        original = tool.get("name") if isinstance(tool, Mapping) else None
        if not isinstance(original, str) or not original or original in mapping:
            raise ValueError("Missing or duplicate tool name")
        base = "".join(
            character if character.isascii() and (character.isalnum() or character in "_-") else "_"
            for character in original
        )
        while "__" in base:
            base = base.replace("__", "_")
        base = base.strip("_") or "tool"
        candidate, suffix_number = base[:64], index + 1
        while candidate in used:
            suffix = f"_{suffix_number}"
            candidate = base[: 64 - len(suffix)] + suffix
            suffix_number += 1
        mapping[original] = candidate
        used.add(candidate)
    return mapping


def normalize_schema(value: Any) -> Any:
    # Normalize only known source aliases and preserve unfamiliar schema fields
    # for forward compatibility.
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result = {key: normalize_schema(item) for key, item in value.items() if not (key == "required" and item is None)}
    if isinstance(result.get("type"), str):
        result["type"] = {"dict": "object", "int": "integer", "float": "number", "bool": "boolean"}.get(
            result["type"], result["type"]
        )
    return result
