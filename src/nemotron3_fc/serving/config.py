"""Configuration for the vLLM server and its API check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron3_fc.paths import resolve_path


@dataclass(frozen=True)
class ServingConfig:
    model_path: Path
    adapter_path: Path
    output_dir: Path
    host: str
    port: int
    base_model_name: str
    adapter_name: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_lora_rank: int
    gpu_memory_utilization: float
    startup_timeout_seconds: int
    request_timeout_seconds: int


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_serving_config(path: Path, *, validate_inputs: bool = True) -> ServingConfig:
    """Load the same configuration for launching or probing a running server."""
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Serving configuration root must be an object")

    base_name = str(raw.get("base_model_name", "")).strip()
    adapter_name = str(raw.get("adapter_name", "")).strip()
    host = str(raw.get("host", "127.0.0.1")).strip()
    if not base_name or not adapter_name or base_name == adapter_name:
        raise ValueError("Set distinct, nonempty base_model_name and adapter_name values")
    if not host:
        raise ValueError("host must be nonempty")

    gpu_fraction = raw.get("gpu_memory_utilization", 0.85)
    if isinstance(gpu_fraction, bool) or not isinstance(gpu_fraction, (int, float)) or not 0 < gpu_fraction < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")

    config = ServingConfig(
        model_path=resolve_path(path.parent, raw["model_path"], "model_path"),
        adapter_path=resolve_path(path.parent, raw["adapter_path"], "adapter_path"),
        output_dir=resolve_path(path.parent, raw["output_dir"], "output_dir"),
        host=host,
        port=_positive_int(raw.get("port", 8000), "port"),
        base_model_name=base_name,
        adapter_name=adapter_name,
        max_model_len=_positive_int(raw.get("max_model_len", 4096), "max_model_len"),
        max_num_seqs=_positive_int(raw.get("max_num_seqs", 8), "max_num_seqs"),
        max_num_batched_tokens=_positive_int(raw.get("max_num_batched_tokens", 8192), "max_num_batched_tokens"),
        max_lora_rank=_positive_int(raw.get("max_lora_rank", 16), "max_lora_rank"),
        gpu_memory_utilization=float(gpu_fraction),
        startup_timeout_seconds=_positive_int(raw.get("startup_timeout_seconds", 1500), "startup_timeout_seconds"),
        request_timeout_seconds=_positive_int(raw.get("request_timeout_seconds", 300), "request_timeout_seconds"),
    )
    if config.port > 65535:
        raise ValueError("port must be at most 65535")

    if validate_inputs:
        required = (
            config.model_path / "config.json",
            config.model_path / "tokenizer_config.json",
            config.adapter_path / "adapter_config.json",
            config.adapter_path / "adapter_model.safetensors",
        )
        missing = [file for file in required if not file.is_file()]
        if missing:
            raise FileNotFoundError("Serving inputs are missing:\n" + "\n".join(map(str, missing)))
        if not any(config.model_path.glob("*.safetensors")):
            raise FileNotFoundError(f"No model safetensors found in {config.model_path}")
    return config
