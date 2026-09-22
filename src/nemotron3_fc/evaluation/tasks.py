"""Prepare assistant-turn prompts from any canonical split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron3_fc.evaluation.config import sha256_file


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item = dict(message)
        if item["role"] == "tool" and not isinstance(item.get("content"), str):
            item["content"] = json.dumps(item["content"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        result.append(item)
    return result


def prepare_tasks(tokenizer: Any, dataset_root: Path, split: str, expected_records: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create one prompt per assistant turn from its recorded preceding history."""
    source = dataset_root / f"{split}.jsonl"
    digest = sha256_file(source)
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("files", {}).get(source.name)
        if entry is not None and entry["sha256"] != digest:
            raise RuntimeError(f"{source.name} differs from its dataset manifest")
    else:
        entry = None
    records = list(read_jsonl(source))
    if not records:
        raise RuntimeError(f"Evaluation split is empty: {source}")
    if expected_records is not None and len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} {split} records, found {len(records)}")
    if entry is not None and len(records) != entry["records"]:
        raise RuntimeError(f"{source.name} record count differs from its manifest")
    if len({row["id"] for row in records}) != len(records):
        raise RuntimeError("Duplicate record IDs in evaluation split")
    tasks = []
    for record_index, row in enumerate(records):
        messages = template_messages(row["messages"])
        count = 0
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            kwargs = {"tools": row["tools"], "tokenize": False, "enable_thinking": False, "truncate_history_thinking": False}
            prefix = tokenizer.apply_chat_template(messages[:message_index], add_generation_prompt=True, **kwargs)
            through = tokenizer.apply_chat_template(messages[:message_index + 1], **kwargs)
            if not through.startswith(prefix):
                raise RuntimeError(f"Native assistant boundary mismatch: {row['id']} at message {message_index}")
            prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            if not prompt_ids:
                raise RuntimeError(f"Empty inference prompt: {row['id']}")
            tasks.append({"task_id": f"{record_index}:{message_index}", "record_id": row["id"], "record_index": record_index, "message_index": message_index, "history": messages[:message_index], "tools": row["tools"], "reference_message": message, "reference_text": through[len(prefix):], "prompt_text": prefix, "prompt_ids": prompt_ids, "multi_turn": sum(item["role"] == "user" for item in messages) > 1, "has_tool_result": any(item["role"] == "tool" for item in messages), "expected_call": bool(message.get("tool_calls"))})
            count += 1
        if not count:
            raise RuntimeError(f"Evaluation record has no assistant turn: {row['id']}")
    info = {"records": len(records), "assistant_turns": len(tasks), "split_sha256": digest, "evaluation_protocol": "assistant_turn_generation_with_reference_history", "max_prompt_tokens": max(len(task["prompt_ids"]) for task in tasks), "call_turns": sum(task["expected_call"] for task in tasks), "no_call_turns": sum(not task["expected_call"] for task in tasks), "multi_turn_records": sum(sum(message["role"] == "user" for message in row["messages"]) > 1 for row in records)}
    return tasks, info
