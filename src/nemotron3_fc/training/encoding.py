"""Native-chat rendering and assistant-only supervision windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EncodedWindow:
    source: str
    record_id: str
    id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    tokens: int
    supervised: int
    no_call: bool
    multi_turn: bool
    has_tool_result: bool


def template_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert structured tool results to the string form expected by chat templates."""
    messages = []
    for message in row["messages"]:
        item = dict(message)
        if item["role"] == "tool" and not isinstance(item.get("content"), str):
            item["content"] = json.dumps(item["content"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        messages.append(item)
    return messages


def render_chat(tokenizer: Any, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        truncate_history_thinking=False,
    )


def encode_record(row: Mapping[str, Any], tokenizer: Any) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Render native chat and label tokens contained wholly inside assistant spans."""
    messages = template_messages(row)
    full_text = render_chat(tokenizer, messages, row["tools"])
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise ValueError(f"Tokenizer ID/offset length mismatch: {row['id']}")
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = render_chat(tokenizer, messages[:index], row["tools"], add_generation_prompt=True)
        through = render_chat(tokenizer, messages[: index + 1], row["tools"])
        if not full_text.startswith(through) or not through.startswith(prefix):
            raise ValueError(f"Assistant boundary mismatch: {row['id']}")
        spans.append((len(prefix), len(through)))
    # Character offsets let us supervise exactly the rendered assistant spans;
    # tool definitions, user turns, and template control tokens remain ignored.
    labels = [
        token if end > start and any(left <= start and end <= right for left, right in spans) else -100
        for token, (start, end) in zip(ids, offsets)
    ]
    if not ids or not any(value != -100 for value in labels):
        raise ValueError(f"No assistant targets: {row['id']}")
    return ids, labels, messages


def record_windows(row: Mapping[str, Any], source: str, tokenizer: Any, *, max_tokens: int, overlap_tokens: int) -> list[EncodedWindow]:
    """Window over-length records while supervising every assistant token exactly once."""
    ids, labels, messages = encode_record(row, tokenizer)
    windows: list[EncodedWindow] = []
    start, previous_end = 0, 0
    no_call = not any(message["role"] == "assistant" and message.get("tool_calls") for message in messages)
    multi_turn = sum(message["role"] == "user" for message in messages) > 1
    has_tool_result = any(message["role"] == "tool" for message in messages)
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        local_labels = labels[start:end].copy()
        if start:
            # Overlap is retained as context, but a target in the overlap was
            # already counted by the preceding window and must not train twice.
            local_labels[: previous_end - start] = [-100] * (previous_end - start)
        local_labels[0] = -100
        supervised = sum(value != -100 for value in local_labels)
        if supervised:
            windows.append(EncodedWindow(
                source=source,
                record_id=str(row["id"]),
                id=f"{row['id']}#w{len(windows)}",
                input_ids=tuple(ids[start:end]),
                labels=tuple(local_labels),
                tokens=end - start,
                supervised=supervised,
                no_call=no_call,
                multi_turn=multi_turn,
                has_tool_result=has_tool_result,
            ))
        previous_end = end
        if end == len(ids):
            break
        start = end - overlap_tokens
    expected = sum(value != -100 for value in labels[1:])
    actual = sum(window.supervised for window in windows)
    if not windows or actual != expected:
        raise ValueError(f"Supervision coverage mismatch: {row['id']} ({actual}/{expected})")
    return windows
