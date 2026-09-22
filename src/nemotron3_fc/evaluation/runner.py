"""Batched vLLM inference with auditable, resumable per-turn outputs."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from nemotron3_fc.evaluation.config import EvaluationConfig, sha256_file
from nemotron3_fc.evaluation.report import paired_comparison, successful_predictions, summarize_adapter, write_json
from nemotron3_fc.evaluation.scoring import BFCL_SOURCE_COMMIT, parse_native_calls_typed, score_generation
from nemotron3_fc.evaluation.tasks import prepare_tasks, read_jsonl

METRIC_VERSION = "bfcl-style-v1"


def _stop_ids(tokenizer: Any) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    vocabulary = tokenizer.get_vocab()
    if "<|im_end|>" in vocabulary:
        ids.add(int(vocabulary["<|im_end|>"]))
    if not ids:
        raise RuntimeError("No assistant stop token found")
    return sorted(ids)


def _probe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = (("single-turn call", lambda t: t["expected_call"] and not t["multi_turn"]), ("no-call", lambda t: not t["expected_call"]), ("multi-turn call", lambda t: t["expected_call"] and t["multi_turn"]))
    selected = []
    for label, predicate in groups:
        matches = [task for task in tasks if predicate(task)]
        if len(matches) < 2:
            raise RuntimeError(f"Need two {label} probe tasks; found {len(matches)}. Set probe=false for a dataset without that category.")
        selected.extend(matches[:2])
    return selected


def _generate(engine: Any, sampling_class: Any, tasks: list[dict[str, Any]], request: Any, stop_ids: list[int], max_tokens: int) -> tuple[list[dict[str, Any]], float]:
    prompts = [{"prompt_token_ids": task["prompt_ids"]} for task in tasks]
    sampling = sampling_class(temperature=0, max_tokens=max_tokens, stop_token_ids=stop_ids, skip_special_tokens=False)
    started = time.monotonic()
    outputs = engine.generate(prompts, sampling, lora_request=request, use_tqdm=False)
    elapsed = time.monotonic() - started
    if len(outputs) != len(tasks):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(tasks)} prompts")
    generations = []
    for task, output in zip(tasks, outputs):
        choice = output.outputs[0]
        ids = list(choice.token_ids)
        generations.append({"generated_text": choice.text, "generated_token_ids": ids, "generated_tokens": len(ids), "prompt_tokens": len(task["prompt_ids"]), "finish_reason": "max_new_tokens" if choice.finish_reason == "length" else "eos", "seconds": elapsed / len(tasks)})
    return generations, elapsed


def _identity(config: EvaluationConfig, adapter_sha: str, info: dict[str, Any], runtime: str) -> dict[str, Any]:
    return {"model_path": str(config.model_path.resolve()), "adapter_sha256": adapter_sha, "split_sha256": info["split_sha256"], "max_new_tokens": config.max_new_tokens, "decoding": "greedy", "protocol": info["evaluation_protocol"], "runtime": runtime}


def _restore(previous: Path | None, destination: Path, identity: dict[str, Any]) -> None:
    if previous is None:
        return
    source_file = previous / destination.name / "predictions.jsonl"
    if not source_file.is_file():
        return
    source_identity = previous / destination.name / "inference-config.json"
    if not source_identity.is_file() or json.loads(source_identity.read_text(encoding="utf-8")) != identity:
        raise RuntimeError(f"Previous results for {destination.name} have a different model, adapter, split, runtime, or generation setting")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "predictions.jsonl"
    if target.exists():
        raise RuntimeError(f"Cannot restore over existing predictions: {target}")
    shutil.copy2(source_file, target)
    write_json(destination / "inference-config.json", identity)


def _rescore(path: Path, tasks: list[dict[str, Any]]) -> None:
    if not path.is_file():
        return
    task_map = {task["task_id"]: task for task in tasks}
    rows = list(read_jsonl(path))
    changed = False
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row["task_id"] not in task_map:
            raise RuntimeError(f"Saved prediction has unknown task ID: {row['task_id']}")
        if row.get("metric_version") != METRIC_VERSION or row.get("metrics", {}).get("bfcl_source_commit") != BFCL_SOURCE_COMMIT:
            row["metrics"] = score_generation(task_map[row["task_id"]], row)
            row["metric_version"] = METRIC_VERSION
            changed = True
    if changed:
        temporary = path.with_suffix(".rescored.jsonl")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Generate, score, and persist each adapter's held-out assistant turns."""
    # Importing vLLM only here keeps scoring and report tests usable on CPU hosts.
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    session_started = time.monotonic()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True)
    tasks, info = prepare_tasks(tokenizer, config.dataset_root, config.split, config.expected_records)
    for task in tasks:
        if task["expected_call"] and not parse_native_calls_typed(task["reference_text"]):
            raise RuntimeError(f"Reference parser mismatch: {task['task_id']}")
    write_json(config.output_dir / "test-info.json", info)
    with (config.output_dir / "test-tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({key: value for key, value in task.items() if key != "prompt_ids"}, ensure_ascii=False, allow_nan=False) + "\n")
    print("FULL SPLIT PREPARED", json.dumps(info), flush=True)
    longest = info["max_prompt_tokens"]
    context = max(4096, longest + config.max_new_tokens)
    engine = LLM(model=str(config.model_path), tokenizer=str(config.model_path), trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1, max_model_len=context, max_num_seqs=config.batch_size, max_num_batched_tokens=max(8192, context), gpu_memory_utilization=config.gpu_memory_utilization, enable_lora=True, max_lora_rank=config.max_lora_rank, max_loras=1, max_cpu_loras=max(2, len(config.adapters)), enforce_eager=True, enable_prefix_caching=False)
    requests = {adapter.name: LoRARequest(adapter.name, index, str(adapter.path)) for index, adapter in enumerate(config.adapters, 1)}
    stop_ids = _stop_ids(tokenizer)
    if config.probe:
        selected = _probe_tasks(tasks)
        base, elapsed = _generate(engine, SamplingParams, selected, None, stop_ids, 64)
        probe = {"task_ids": [task["task_id"] for task in selected], "base": base}
        print("PROBE | base | seconds:", round(elapsed, 2), flush=True)
        for adapter in config.adapters:
            outputs, elapsed = _generate(engine, SamplingParams, selected, requests[adapter.name], stop_ids, 64)
            changed = sum(a["generated_token_ids"] != b["generated_token_ids"] for a, b in zip(base, outputs))
            probe[adapter.name] = outputs
            print("PROBE | adapter:", adapter.name, "| changed:", changed, "/", len(selected), "| seconds:", round(elapsed, 2), flush=True)
            if changed == 0:
                raise RuntimeError(f"Adapter {adapter.name} did not alter any probe output")
        write_json(config.output_dir / "vllm-adapter-probe.json", probe)
    runtime = f"vllm-{vllm.__version__}"
    summaries = {}
    for adapter in config.adapters:
        root = config.output_dir / adapter.name
        root.mkdir(parents=True, exist_ok=True)
        predictions = root / "predictions.jsonl"
        identity_file = root / "inference-config.json"
        digest = sha256_file(adapter.path / "adapter_model.safetensors")
        identity = _identity(config, digest, info, runtime)
        if predictions.is_file() and not identity_file.is_file():
            raise RuntimeError(f"Cannot resume {predictions} without its inference-config.json identity")
        if identity_file.is_file() and json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"Existing predictions for {adapter.name} use different inputs or settings")
        if not predictions.exists():
            _restore(config.previous_results_dir, root, identity)
        write_json(identity_file, identity)
        _rescore(predictions, tasks)
        successful = successful_predictions(predictions)
        expected_ids = {task["task_id"] for task in tasks}
        if not set(successful).issubset(expected_ids):
            raise RuntimeError("Saved predictions contain unknown test tasks")
        pending = [task for task in tasks if task["task_id"] not in successful]
        status = "completed"
        print("ADAPTER START |", adapter.name, "| completed:", len(successful), "| pending:", len(pending), flush=True)
        with predictions.open("a", encoding="utf-8") as handle:
            for offset in range(0, len(pending), config.batch_size):
                if config.session_limit_seconds is not None and time.monotonic() - session_started >= config.session_limit_seconds:
                    status = "session_time_limit"
                    break
                batch = pending[offset:offset + config.batch_size]
                generated, elapsed = _generate(engine, SamplingParams, batch, requests[adapter.name], stop_ids, config.max_new_tokens)
                for task, generation in zip(batch, generated):
                    row = {"status": "ok", "adapter": adapter.name, "task_id": task["task_id"], "record_id": task["record_id"], "record_index": task["record_index"], "message_index": task["message_index"], "multi_turn": task["multi_turn"], "has_tool_result": task["has_tool_result"], "reference_message": task["reference_message"], "reference_text": task["reference_text"], **generation, "metric_version": METRIC_VERSION, "metrics": score_generation(task, generation)}
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    successful[task["task_id"]] = row
                handle.flush()
                os.fsync(handle.fileno())
                print("BATCH |", adapter.name, "| turns:", len(successful), "/", len(tasks), "| seconds:", round(elapsed, 2), "| generated tokens:", sum(row["generated_tokens"] for row in generated), flush=True)
        summary = summarize_adapter(adapter.name, adapter.step, successful, status, info, digest, runtime)
        write_json(root / "summary.json", summary)
        summaries[adapter.name] = summary
        print("ADAPTER SUMMARY", json.dumps(summary), flush=True)
        if status != "completed":
            break
    if len(config.adapters) == 2:
        comparison = paired_comparison(config.output_dir, tasks, config.adapters[0].name, config.adapters[1].name)
        print("PAIRED COMPARISON", json.dumps(comparison), flush=True)
    return summaries
