"""Validate base chat, adapter selection, and one complete tool exchange."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nemotron3_fc.serving.config import ServingConfig


def _api_url(config: ServingConfig, suffix: str) -> str:
    return f"http://{config.host}:{config.port}{suffix}"


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}:\n{body}") from error


def _chat(config: ServingConfig, payload: dict[str, Any]) -> dict[str, Any]:
    payload.update(
        {
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False, "truncate_history_thinking": False},
        }
    )
    return _post_json(_api_url(config, "/v1/chat/completions"), payload, config.request_timeout_seconds)


def _message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        return response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Invalid chat completion response: {response}") from error


def weather_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Return the current weather for a requested location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    }


def execute_weather_fixture(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic tool data so this test never depends on live weather."""
    if arguments != {"location": "Colombo, Sri Lanka", "unit": "celsius"}:
        raise RuntimeError(f"Unexpected tool arguments: {arguments}")
    return {
        "location": arguments["location"],
        "temperature": 29,
        "unit": arguments["unit"],
        "condition": "partly cloudy",
        "source": "serving-validation-fixture",
    }


def validate_api(config: ServingConfig) -> dict[str, Any]:
    """Exercise the same API calls that passed in the serving notebook."""
    started = time.monotonic()
    models = _get_json(_api_url(config, "/v1/models"), config.request_timeout_seconds)
    model_ids = {item["id"] for item in models.get("data", [])}
    required_ids = {config.base_model_name, config.adapter_name}
    if not required_ids.issubset(model_ids):
        raise RuntimeError(f"Missing API model IDs: {sorted(required_ids - model_ids)}")
    print("MODELS", sorted(model_ids), flush=True)

    base_response = _chat(
        config,
        {
            "model": config.base_model_name,
            "messages": [{"role": "user", "content": "In one short sentence, confirm that the server is ready."}],
            "max_tokens": 64,
        },
    )
    base_text = _message(base_response).get("content") or ""
    if not base_text.strip():
        raise RuntimeError("Base model returned no chat content")
    print("BASE CHAT", base_text.strip(), flush=True)

    tool = weather_tool()
    user_message = {
        "role": "user",
        "content": "Use the available tool to obtain the current weather in Colombo, Sri Lanka. Use celsius.",
    }
    call_response = _chat(
        config,
        {
            "model": config.adapter_name,
            "messages": [user_message],
            "tools": [tool],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": 256,
        },
    )
    assistant_message = _message(call_response)
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1:
        raise RuntimeError(f"Expected one adapter tool call, received {len(calls)}: {call_response}")
    call = calls[0]
    if call.get("function", {}).get("name") != "get_current_weather" or not call.get("id"):
        raise RuntimeError(f"Unexpected adapter tool call: {call}")
    try:
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid tool arguments: {call}") from error
    tool_result = execute_weather_fixture(arguments)
    print("ADAPTER TOOL CALL", json.dumps(call, ensure_ascii=False), flush=True)

    final_response = _chat(
        config,
        {
            "model": config.adapter_name,
            "messages": [
                user_message,
                {"role": "assistant", "content": assistant_message.get("content"), "tool_calls": calls},
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "get_current_weather",
                    "content": json.dumps(tool_result),
                },
            ],
            "tools": [tool],
            "tool_choice": "none",
            "max_tokens": 128,
        },
    )
    final_text = _message(final_response).get("content") or ""
    if not final_text.strip():
        raise RuntimeError("Adapter returned no final response after the tool result")
    print("FINAL RESPONSE", final_text.strip(), flush=True)

    return {
        "status": "passed",
        "available_models": sorted(model_ids),
        "base_response": base_text,
        "tool_call": call,
        "tool_result": tool_result,
        "final_response": final_text,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_serving_check(config: ServingConfig) -> dict[str, Any]:
    """Write a report even if an API check fails."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "serving-check.json"
    try:
        report = validate_api(config)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SERVING CHECK PASSED | report:", report_path, flush=True)
    return report
