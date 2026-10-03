"""Measure complete tool exchanges through the OpenAI-compatible API."""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemotron3_fc.serving.check import weather_tool
from nemotron3_fc.serving.config import ServingConfig


@dataclass(frozen=True)
class WeatherCase:
    location: str
    unit: str


@dataclass(frozen=True)
class BenchmarkConfig:
    concurrency_levels: tuple[int, ...]
    requests_per_level: int
    warmup_requests_per_model: int
    tool_max_tokens: int
    final_max_tokens: int
    gpu_index: int
    cases: tuple[WeatherCase, ...]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Read and validate the workload declared beside the serving settings."""
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("benchmark")
    if not isinstance(raw, dict):
        raise ValueError("Add a benchmark object to the serving configuration")

    levels = raw.get("concurrency_levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("concurrency_levels must be a nonempty list")
    parsed_levels = tuple(_positive_int(value, "concurrency level") for value in levels)
    if len(set(parsed_levels)) != len(parsed_levels):
        raise ValueError("concurrency_levels must not contain duplicates")

    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    parsed_cases = []
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Each case must be an object")
        location = item.get("location")
        unit = item.get("unit")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Each case needs a nonempty location")
        if unit not in {"celsius", "fahrenheit"}:
            raise ValueError("Each case unit must be celsius or fahrenheit")
        parsed_cases.append(WeatherCase(location.strip(), unit))

    warmup = raw.get("warmup_requests_per_model", 1)
    gpu_index = raw.get("gpu_index", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_requests_per_model must be a nonnegative integer")
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a nonnegative integer")

    return BenchmarkConfig(
        concurrency_levels=parsed_levels,
        requests_per_level=_positive_int(raw.get("requests_per_level", 8), "requests_per_level"),
        warmup_requests_per_model=warmup,
        tool_max_tokens=_positive_int(raw.get("tool_max_tokens", 256), "tool_max_tokens"),
        final_max_tokens=_positive_int(raw.get("final_max_tokens", 128), "final_max_tokens"),
        gpu_index=gpu_index,
        cases=tuple(parsed_cases),
    )


def _chat_payload(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                  tool_choice: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False, "truncate_history_thinking": False},
    }


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:1000]}") from error


def _stream_final(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Time the first visible content delta and collect final token usage."""
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.monotonic()
    first_content = None
    content = []
    usage = None
    finished = False
    finish_reason = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    finished = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"Streaming response error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    fragment = (choice.get("delta") or {}).get("content")
                    if fragment:
                        if first_content is None:
                            first_content = time.monotonic() - started
                        content.append(fragment)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:1000]}") from error

    if not finished or first_content is None or not "".join(content).strip():
        raise RuntimeError("Final streamed response had no complete visible answer")
    return {
        "first_content_seconds": first_content,
        "duration_seconds": time.monotonic() - started,
        "usage": usage,
        "finish_reason": finish_reason,
    }


def _token_count(usage: Any, field: str) -> int | None:
    value = usage.get(field) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _run_case(config: ServingConfig, workload: BenchmarkConfig, model: str,
              case: WeatherCase, concurrency: int, index: int) -> dict[str, Any]:
    """Run a model tool call, a local fixture, and the final model response."""
    row: dict[str, Any] = {
        "model": model, "concurrency": concurrency, "request_index": index,
        "location": case.location, "unit": case.unit, "status": "failed",
    }
    started = time.monotonic()
    url = f"http://{config.host}:{config.port}/v1/chat/completions"
    tools = [weather_tool()]
    user = {
        "role": "user",
        "content": f"Use the available tool to obtain the current weather in {case.location}. Use {case.unit}.",
    }

    try:
        tool_started = time.monotonic()
        call_response = _post_json(
            url,
            _chat_payload(model, [user], tools, "required", workload.tool_max_tokens),
            config.request_timeout_seconds,
        )
        row["tool_call_seconds"] = time.monotonic() - tool_started
        assistant = call_response["choices"][0]["message"]
        calls = assistant.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError(f"Expected one tool call, received {len(calls)}")
        call = calls[0]
        if call.get("function", {}).get("name") != "get_current_weather" or not call.get("id"):
            raise ValueError("Tool call has an unexpected name or no ID")
        arguments = json.loads(call["function"]["arguments"])
        if not isinstance(arguments, dict) or not isinstance(arguments.get("location"), str):
            raise ValueError("Tool arguments have no location")
        if arguments.get("unit") not in {"celsius", "fahrenheit"}:
            raise ValueError("Tool arguments have an invalid unit")
        row["arguments_match_request"] = arguments == asdict(case)
        tool_result = {
            "location": arguments["location"], "unit": arguments["unit"],
            "temperature": 29, "condition": "partly cloudy", "source": "serving-benchmark-fixture",
        }

        final_messages = [
            user,
            {"role": "assistant", "content": assistant.get("content"), "tool_calls": calls},
            {"role": "tool", "tool_call_id": call["id"], "name": "get_current_weather",
             "content": json.dumps(tool_result)},
        ]
        final = _stream_final(
            url,
            _chat_payload(model, final_messages, tools, "none", workload.final_max_tokens),
            config.request_timeout_seconds,
        )
        row["first_final_content_seconds"] = final["first_content_seconds"]
        row["final_response_seconds"] = final["duration_seconds"]
        row["final_at_token_limit"] = final["finish_reason"] == "length"
        row["tool_prompt_tokens"] = _token_count(call_response.get("usage"), "prompt_tokens")
        row["final_prompt_tokens"] = _token_count(final["usage"], "prompt_tokens")
        row["tool_completion_tokens"] = _token_count(call_response.get("usage"), "completion_tokens")
        row["final_completion_tokens"] = _token_count(final["usage"], "completion_tokens")
        if row["tool_prompt_tokens"] is not None and row["final_prompt_tokens"] is not None:
            row["prompt_tokens"] = row["tool_prompt_tokens"] + row["final_prompt_tokens"]
        else:
            row["prompt_tokens"] = None
        if row["tool_completion_tokens"] is not None and row["final_completion_tokens"] is not None:
            row["completion_tokens"] = row["tool_completion_tokens"] + row["final_completion_tokens"]
        else:
            row["completion_tokens"] = None
        row["status"] = "completed"
    except (KeyError, IndexError, TypeError, ValueError, OSError, TimeoutError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"[:1200]

    row["total_seconds"] = time.monotonic() - started
    return row


def _gpu_memory_used_mib(index: int) -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=5, check=True,
        )
        return int(result.stdout.strip().splitlines()[0].strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _sample_memory(index: int, stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        measured = _gpu_memory_used_mib(index)
        if measured is not None:
            samples.append(measured)
        stop.wait(1)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def _summarize(rows: list[dict[str, Any]], model: str, concurrency: int,
               wall_seconds: float, memory_samples: list[int]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    token_values = [row["completion_tokens"] for row in completed]
    total_tokens = sum(token_values) if all(value is not None for value in token_values) else None
    latencies = [row["total_seconds"] for row in completed]
    tool_latencies = [row["tool_call_seconds"] for row in completed]
    final_latencies = [row["final_response_seconds"] for row in completed]
    first_output = [row["first_final_content_seconds"] for row in completed]

    return {
        "model": model,
        "concurrency": concurrency,
        "requests": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "arguments_matching_request": sum(row.get("arguments_match_request") is True for row in completed),
        "outputs_at_token_limit": sum(row.get("final_at_token_limit") is True for row in completed),
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_minute": round(len(completed) * 60 / wall_seconds, 3),
        "completion_tokens": total_tokens,
        "completion_tokens_per_second": round(total_tokens / wall_seconds, 3) if total_tokens is not None else None,
        "total_latency_p50_seconds": _percentile(latencies, 0.5),
        "total_latency_p95_seconds": _percentile(latencies, 0.95),
        "tool_call_latency_p50_seconds": _percentile(tool_latencies, 0.5),
        "tool_call_latency_p95_seconds": _percentile(tool_latencies, 0.95),
        "final_response_latency_p50_seconds": _percentile(final_latencies, 0.5),
        "final_response_latency_p95_seconds": _percentile(final_latencies, 0.95),
        "first_final_content_p50_seconds": _percentile(first_output, 0.5),
        "first_final_content_p95_seconds": _percentile(first_output, 0.95),
        "gpu_memory_used_peak_mib": max(memory_samples) if memory_samples else None,
        "gpu_memory_used_min_mib": min(memory_samples) if memory_samples else None,
    }


def _write_results(directory: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "model", "concurrency", "request_index", "location", "unit", "status", "error",
        "arguments_match_request", "tool_call_seconds", "first_final_content_seconds",
        "final_response_seconds", "total_seconds", "final_at_token_limit",
        "tool_prompt_tokens", "final_prompt_tokens", "prompt_tokens",
        "tool_completion_tokens", "final_completion_tokens", "completion_tokens",
    ]
    with (directory / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(config: ServingConfig, workload: BenchmarkConfig) -> dict[str, Any]:
    """Warm both models, run each concurrency level, and persist every result."""
    run_name = datetime.now(timezone.utc).strftime("benchmark-%Y%m%d-%H%M%S-%f")
    output = config.output_dir / run_name
    models = (config.base_model_name, config.adapter_name)
    report: dict[str, Any] = {
        "status": "running",
        "models": models,
        "workload": asdict(workload),
        "metric_notes": {
            "first_final_content": "Time from the final API request to its first visible content chunk",
            "completion_tokens_per_second": "Both calls' completion tokens divided by level wall time",
            "gpu_memory_used": "Total device memory reported by nvidia-smi, sampled once per second",
        },
        "levels": [],
    }
    all_rows: list[dict[str, Any]] = []
    _write_results(output, report, all_rows)
    print("BENCHMARK OUTPUT", output, flush=True)

    for model in models:
        for index in range(workload.warmup_requests_per_model):
            row = _run_case(config, workload, model, workload.cases[index % len(workload.cases)], 1, -1)
            if row["status"] != "completed":
                report["status"] = "failed_during_warmup"
                report["warmup_error"] = row.get("error")
                _write_results(output, report, all_rows)
                raise RuntimeError(f"Warmup failed for {model}: {row.get('error')}")
        print("WARMUP COMPLETE", model, flush=True)

    for concurrency in workload.concurrency_levels:
        for model in models:
            samples: list[int] = []
            stop = threading.Event()
            sampler = threading.Thread(target=_sample_memory, args=(workload.gpu_index, stop, samples), daemon=True)
            level_rows = []
            started = time.monotonic()
            sampler.start()
            try:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            _run_case, config, workload, model,
                            workload.cases[index % len(workload.cases)], concurrency, index,
                        )
                        for index in range(workload.requests_per_level)
                    ]
                    for future in as_completed(futures):
                        level_rows.append(future.result())
            finally:
                ended = time.monotonic()
                stop.set()
                sampler.join(timeout=6)
            wall_seconds = ended - started
            level_rows.sort(key=lambda row: row["request_index"])
            summary = _summarize(level_rows, model, concurrency, wall_seconds, samples)
            all_rows.extend(level_rows)
            report["levels"].append(summary)
            _write_results(output, report, all_rows)
            print("BENCHMARK LEVEL", json.dumps(summary), flush=True)

    report["status"] = "completed_with_failures" if any(row["status"] != "completed" for row in all_rows) else "completed"
    _write_results(output, report, all_rows)
    print("BENCHMARK COMPLETE | summary:", output / "summary.json", flush=True)
    print("BENCHMARK REQUESTS:", output / "requests.csv", flush=True)
    return report
