"""Launch and supervise the vLLM OpenAI-compatible API process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from nemotron3_fc.serving.config import ServingConfig


def server_command(config: ServingConfig) -> list[str]:
    """Build the vLLM 0.18.0 command validated on the RTX PRO 6000."""
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config.model_path),
        "--tokenizer",
        str(config.model_path),
        "--served-model-name",
        config.base_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank",
        str(config.max_lora_rank),
        "--lora-modules",
        f"{config.adapter_name}={config.adapter_path}",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--disable-frontend-multiprocessing",
    ]


def _models_url(config: ServingConfig) -> str:
    return f"http://{config.host}:{config.port}/v1/models"


def _log_tail(path: Path, size: int = 8000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-size:] if path.is_file() else "No server log found"


def wait_for_ready(process: subprocess.Popen, config: ServingConfig, log_path: Path) -> None:
    """Poll the API until ready while exposing startup progress and early failures."""
    started = time.monotonic()
    deadline = started + config.startup_timeout_seconds
    next_progress = started

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}:\n{_log_tail(log_path)}")
        try:
            with urllib.request.urlopen(_models_url(config), timeout=10) as response:
                payload = json.load(response)
            names = {item.get("id") for item in payload.get("data", [])}
            expected = {config.base_model_name, config.adapter_name}
            if not expected.issubset(names):
                raise RuntimeError(f"Server is missing model IDs {sorted(expected - names)}")
            print(f"SERVER READY after {(time.monotonic() - started) / 60:.1f} minutes", flush=True)
            print("AVAILABLE MODELS", sorted(names), flush=True)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass

        if time.monotonic() >= next_progress:
            print(f"WAITING FOR VLLM {(time.monotonic() - started) / 60:.1f} minutes", flush=True)
            next_progress = time.monotonic() + 60
        time.sleep(10)

    raise TimeoutError(f"vLLM startup timed out:\n{_log_tail(log_path)}")


def stop_server(process: subprocess.Popen) -> None:
    """Stop the API and its engine worker after an interrupt or startup failure."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=30)


def run_server(config: ServingConfig, *, verify: bool = False) -> int:
    """Serve continuously, or validate the API and then stop it."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "vllm-server.log"
    print("SERVER LOG", log_path, flush=True)
    print("BASE MODEL", config.base_model_name, flush=True)
    print("LORA ADAPTER", config.adapter_name, config.adapter_path, flush=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            server_command(config),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        print("SERVER PID", process.pid, flush=True)
        try:
            wait_for_ready(process, config, log_path)
            if verify:
                from nemotron3_fc.serving.check import run_serving_check

                run_serving_check(config)
                return 0
            return process.wait()
        except KeyboardInterrupt:
            print("SERVER SHUTDOWN requested", flush=True)
            return 130 if verify else 0
        finally:
            stop_server(process)
            print("SERVER STOPPED | log:", log_path, flush=True)
