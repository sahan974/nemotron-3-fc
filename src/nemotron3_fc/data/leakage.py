"""Duplicate fingerprints and cross-split leakage keys."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable, Mapping

from nemotron3_fc.data.schema import CanonicalRecord


def record_fingerprint(record: CanonicalRecord) -> str:
    """Hash semantic content while excluding record IDs and source metadata."""
    messages = []
    for message in record.messages:
        if message.role == "assistant" and message.tool_calls:
            messages.append({"role": "assistant", "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in message.tool_calls]})
        elif message.role == "tool":
            messages.append({"role": "tool", "name": message.name, "content": message.content})
        else:
            messages.append({"role": message.role, "content": message.content})
    return _hash_json({"tools": record.tools, "messages": messages})


def user_query_keys(record: CanonicalRecord) -> set[str]:
    return {" ".join(message.content.casefold().split()) for message in record.messages if message.role == "user" and message.content.strip()}


def offered_tool_keys(record: CanonicalRecord) -> set[str]:
    return {_hash_json(tool["function"]) for tool in record.tools}


def numeric_query_template(text: str) -> str:
    tokens: list[str] = []
    current: list[str] = []
    current_kind = None

    def finish() -> None:
        if current:
            token = "".join(current)
            tokens.append("<number>" if any(character.isdigit() for character in token) else token.casefold())
            current.clear()

    for character in text:
        if character.isascii() and (character.isalnum() or character == "_"):
            kind = "ascii"
        elif character.isdigit():
            kind = "digit"
        elif character.isalpha():
            kind = "unicode_letter"
        else:
            kind = None
        if kind != current_kind:
            finish()
        if kind is not None:
            current.append(character)
        current_kind = kind
    finish()
    return " ".join(tokens)


def query_template_keys(record: CanonicalRecord) -> set[str]:
    templates = set()
    for message in record.messages:
        if message.role != "user":
            continue
        template = numeric_query_template(message.content)
        tokens = template.split()
        if len(tokens) >= 4 and sum(token != "<number>" for token in tokens) >= 2:
            templates.add(template)
    return templates


def record_statistics(records: Iterable[CanonicalRecord]) -> dict[str, int]:
    records = list(records)
    return {
        "records": len(records),
        "multi_turn_records": sum(is_multiturn(record) for record in records),
        "without_calls": sum(not any(message.tool_calls for message in record.messages) for record in records),
        "with_tool_results": sum(any(message.role == "tool" for message in record.messages) for record in records),
    }


def audit_split_leakage(split_records: Mapping[str, Mapping[str, list[CanonicalRecord]]]) -> dict:
    """Validate identities/content and prove configured leakage keys do not cross splits."""
    owners: dict[str, dict[str, str]] = {"exact_user_query": {}, "offered_tool_definition": {}, "query_template": {}}
    seen_ids, seen_fingerprints = set(), set()
    counts = {}
    for source, splits in split_records.items():
        counts[source] = {}
        for split, records in splits.items():
            counts[source][split] = record_statistics(records)
            for record in records:
                record.validate()
                if record.source != source or record.id in seen_ids:
                    raise RuntimeError(f"Wrong source or repeated record ID: {record.id}")
                seen_ids.add(record.id)
                fingerprint = record_fingerprint(record)
                if fingerprint in seen_fingerprints:
                    raise RuntimeError(f"Exact duplicate retained: {record.id}")
                seen_fingerprints.add(fingerprint)
                key_sets = {
                    "exact_user_query": user_query_keys(record),
                    "offered_tool_definition": offered_tool_keys(record),
                    "query_template": query_template_keys(record),
                }
                for category, keys in key_sets.items():
                    for key in keys:
                        previous = owners[category].get(key)
                        if previous is not None and previous != split:
                            raise RuntimeError(f"{category} crosses splits: {record.id} ({previous} -> {split})")
                        owners[category][key] = split
    return {
        "counts": counts,
        "unique_records": len(seen_ids),
        "unique_exact_contents": len(seen_fingerprints),
        "cross_split_exact_queries": 0,
        "cross_split_offered_tool_definitions": 0,
        "cross_split_query_templates": 0,
    }


def is_multiturn(record: CanonicalRecord) -> bool:
    return sum(message.role == "user" for message in record.messages) > 1


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
