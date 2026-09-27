import json
from pathlib import Path

import pytest

from nemotron3_fc.serving.config import load_serving_config
from nemotron3_fc.serving.server import run_server, server_command
from nemotron3_fc.serving.check import run_serving_check


def serving_fixture(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    for file in (model / "config.json", model / "tokenizer_config.json", model / "weights.safetensors"):
        file.touch()
    for file in (adapter / "adapter_config.json", adapter / "adapter_model.safetensors"):
        file.touch()
    config_path = tmp_path / "serve.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(model),
                "adapter_path": str(adapter),
                "output_dir": str(tmp_path / "reports"),
                "base_model_name": "nemotron-3-fc",
                "adapter_name": "epoch-1",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_serving_command_uses_validated_parser_and_adapter(tmp_path: Path) -> None:
    config = load_serving_config(serving_fixture(tmp_path))
    command = server_command(config)
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--lora-modules") + 1] == f"epoch-1={config.adapter_path}"
    assert "--enable-auto-tool-choice" in command
    assert "--enforce-eager" in command


def test_serving_check_exercises_complete_tool_exchange(tmp_path: Path, monkeypatch) -> None:
    config = load_serving_config(serving_fixture(tmp_path))
    requests = []

    def fake_get(url, timeout):
        return {"data": [{"id": "nemotron-3-fc"}, {"id": "epoch-1"}]}

    def fake_post(url, payload, timeout):
        requests.append(payload)
        if len(requests) == 1:
            return {"choices": [{"message": {"content": "The server is ready."}}]}
        if len(requests) == 2:
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location":"Colombo, Sri Lanka","unit":"celsius"}',
                            },
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Colombo is 29°C and partly cloudy."}}]}

    monkeypatch.setattr("nemotron3_fc.serving.check._get_json", fake_get)
    monkeypatch.setattr("nemotron3_fc.serving.check._post_json", fake_post)
    report = run_serving_check(config)

    assert report["status"] == "passed"
    assert [request["model"] for request in requests] == ["nemotron-3-fc", "epoch-1", "epoch-1"]
    assert requests[1]["tool_choice"] == "required"
    assert requests[2]["messages"][-1]["role"] == "tool"
    assert json.loads(requests[2]["messages"][-1]["content"])["temperature"] == 29
    assert (config.output_dir / "serving-check.json").is_file()


def test_serving_config_rejects_missing_adapter_weights(tmp_path: Path) -> None:
    config_path = serving_fixture(tmp_path)
    (tmp_path / "adapter" / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load_serving_config(config_path)


def test_verify_serving_stops_server_after_check(tmp_path: Path, monkeypatch) -> None:
    config = load_serving_config(serving_fixture(tmp_path))
    events = []

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr("nemotron3_fc.serving.server.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("nemotron3_fc.serving.server.wait_for_ready", lambda *args: events.append("ready"))
    monkeypatch.setattr("nemotron3_fc.serving.check.run_serving_check", lambda *args: events.append("check"))
    monkeypatch.setattr("nemotron3_fc.serving.server.stop_server", lambda *args: events.append("stopped"))

    assert run_server(config, verify=True) == 0
    assert events == ["ready", "check", "stopped"]
