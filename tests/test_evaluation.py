"""CPU checks for inference preparation, scoring, and paired reporting."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from nemotron3_fc.evaluation.config import load_evaluation_config, sha256_file
from nemotron3_fc.evaluation.report import paired_comparison, summarize_adapter
from nemotron3_fc.evaluation.scoring import parse_native_calls_typed, score_generation
from nemotron3_fc.evaluation.tasks import prepare_tasks
from nemotron3_fc.evaluation.runner import run_evaluation


TOOL = {"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def call(city: str) -> str:
    return f"<tool_call><function=weather><parameter=city>{json.dumps(city)}</parameter></function></tool_call>"


def task(reference: str, expected: bool = True) -> dict:
    return {"task_id": "0:1", "reference_text": reference, "expected_call": expected, "tools": [TOOL]}


def test_ast_accepts_normalized_string_but_strict_metric_remains_exact():
    result = score_generation(task(call("New-York")), {"generated_text": call("new york")})
    assert result["bfcl_ast_correct"] is True
    assert result["native_calls_exact"] is False
    assert result["bfcl_relevance_correct"] is True


def test_ast_rejects_extra_parameter_and_malformed_call():
    extra = '<tool_call><function=weather><parameter=city>"Paris"</parameter><parameter=unit>"C"</parameter></function></tool_call>'
    assert score_generation(task(call("Paris")), {"generated_text": extra})["bfcl_ast_error_type"] == "simple_function_checker:unexpected_param"
    malformed = score_generation(task(call("Paris")), {"generated_text": "<tool_call><function=weather>"})
    assert malformed["bfcl_decode_valid"] is False
    assert malformed["bfcl_ast_correct"] is False
    with pytest.raises(ValueError, match="duplicate_parameter"):
        parse_native_calls_typed('<tool_call><function=weather><parameter=city>"A"</parameter><parameter=city>"B"</parameter></function></tool_call>')


def test_no_call_irrelevance_and_parallel_order_independence():
    assert score_generation(task("Hello", False), {"generated_text": "Hello"})["bfcl_irrelevance_correct"] is True
    assert score_generation(task("Hello", False), {"generated_text": call("Paris")})["bfcl_irrelevance_correct"] is False
    other = {"type": "function", "function": {"name": "time", "parameters": {"type": "object", "properties": {}, "required": []}}}
    time_call = "<tool_call><function=time></function></tool_call>"
    multi = {**task(call("Paris") + time_call), "tools": [TOOL, other]}
    assert score_generation(multi, {"generated_text": time_call + call("Paris")})["bfcl_ast_correct"] is True


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, enable_thinking, truncate_history_thinking, add_generation_prompt=False):
        rendered = "".join(f"{message['role']}:{call(message['tool_calls'][0]['function']['arguments']['city']) if message.get('tool_calls') else message.get('content') or ''}|" for message in messages)
        return rendered + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(text.encode())}


def test_prepare_tasks_checks_manifest_and_assistant_boundaries(tmp_path: Path):
    row = {"id": "example", "source": "sample", "tools": [TOOL], "messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]}]}
    split = tmp_path / "test.jsonl"
    split.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"test.jsonl": {"sha256": sha256_file(split), "records": 1}}}), encoding="utf-8")
    tasks, info = prepare_tasks(FakeTokenizer(), tmp_path, "test", 1)
    assert info["assistant_turns"] == 1
    assert tasks[0]["expected_call"] is True
    assert tasks[0]["history"] == row["messages"][:1]
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"test.jsonl": {"sha256": "bad", "records": 1}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest"):
        prepare_tasks(FakeTokenizer(), tmp_path, "test", 1)


def test_config_and_paired_summary(tmp_path: Path):
    model = tmp_path / "model"
    adapter1 = tmp_path / "adapter-1"
    adapter2 = tmp_path / "adapter-2"
    for directory in (model, adapter1, adapter2):
        directory.mkdir()
    (model / "config.json").write_text("{}")
    for adapter in (adapter1, adapter2):
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "test.jsonl").write_text("{}\n")
    raw = {"model_path": str(model), "dataset_root": str(tmp_path), "output_dir": str(tmp_path / "out"), "adapters": [{"name": "first", "path": str(adapter1)}, {"name": "second", "path": str(adapter2)}]}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw))
    assert len(load_evaluation_config(config_file).adapters) == 2
    output = tmp_path / "out"
    for name, correct in (("first", True), ("second", False)):
        directory = output / name
        directory.mkdir(parents=True)
        metrics = score_generation(task(call("Paris")), {"generated_text": call("Paris") if correct else call("London")})
        row = {"status": "ok", "task_id": "0:1", "record_id": "example", "message_index": 1, "metrics": metrics, "seconds": 1, "generated_tokens": 8, "finish_reason": "eos"}
        (directory / "predictions.jsonl").write_text(json.dumps(row) + "\n")
        summary = summarize_adapter(name, None, {"0:1": row}, "completed", {"records": 1, "assistant_turns": 1, "split_sha256": "hash"}, "adapter-hash", "vllm-test")
        assert summary["successful_assistant_turns"] == 1
    comparison = paired_comparison(output, [{"task_id": "0:1", "record_id": "example", "message_index": 1, "expected_call": True}], "first", "second")
    assert comparison["first_only_bfcl_ast_correct"] == 1
    assert (output / "paired-comparison.csv").is_file()


def test_runner_generates_and_resumes_without_repeating_calls(tmp_path: Path, monkeypatch):
    model, adapter, data = (tmp_path / item for item in ("model", "adapter", "data"))
    for directory in (model, adapter, data):
        directory.mkdir()
    (model / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    row = {"id": "example", "source": "sample", "tools": [TOOL], "messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]}]}
    (data / "test.jsonl").write_text(json.dumps(row) + "\n")
    raw = {"model_path": str(model), "dataset_root": str(data), "output_dir": str(tmp_path / "out"), "adapters": [{"name": "first", "path": str(adapter)}], "probe": False}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw))

    class Tokenizer:
        eos_token_id = 1

        def get_vocab(self):
            return {"<|im_end|>": 2}

        def apply_chat_template(self, messages, *, tools, tokenize, enable_thinking, truncate_history_thinking, add_generation_prompt=False):
            rendered = "".join("user:" + message["content"] + "<|im_end|>" if message["role"] == "user" else "assistant:" + call(message["tool_calls"][0]["function"]["arguments"]["city"]) + "<|im_end|>" for message in messages)
            return rendered + ("assistant:" if add_generation_prompt else "")

        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": list(text.encode())}

    class Engine:
        calls = 0

        def __init__(self, **kwargs):
            assert kwargs["enable_lora"] is True

        def generate(self, prompts, sampling, *, lora_request, use_tqdm):
            Engine.calls += 1
            assert lora_request is not None
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=call("Paris"), token_ids=[10, 11], finish_reason="stop")]) for _ in prompts]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "test"
    fake_vllm.LLM = Engine
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    fake_lora = types.ModuleType("vllm.lora")
    fake_request = types.ModuleType("vllm.lora.request")
    fake_request.LoRARequest = lambda *args: args
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    for name, module in (("vllm", fake_vllm), ("vllm.lora", fake_lora), ("vllm.lora.request", fake_request), ("transformers", fake_transformers)):
        monkeypatch.setitem(sys.modules, name, module)

    config = load_evaluation_config(config_file)
    first = run_evaluation(config)
    assert first["first"]["primary_bfcl_ast_accuracy"] == 1.0
    assert Engine.calls == 1
    second = run_evaluation(config)
    assert second["first"]["successful_assistant_turns"] == 1
    assert Engine.calls == 1
    identity = tmp_path / "out" / "first" / "inference-config.json"
    identity.unlink()
    with pytest.raises(RuntimeError, match="without its inference-config"):
        run_evaluation(config)
