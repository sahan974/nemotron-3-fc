"""Configuration and input identity for turn-level inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    path: Path
    step: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    dataset_root: Path
    split: str
    output_dir: Path
    adapters: tuple[AdapterConfig, ...]
    expected_records: int | None
    batch_size: int
    max_new_tokens: int
    gpu_memory_utilization: float
    max_lora_rank: int
    session_limit_seconds: int | None
    previous_results_dir: Path | None
    probe: bool


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Validate explicit paths and resource settings before loading the model."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    adapters = tuple(AdapterConfig(str(item["name"]), Path(item["path"]), item.get("step")) for item in raw["adapters"])
    config = EvaluationConfig(
        model_path=Path(raw["model_path"]), dataset_root=Path(raw["dataset_root"]),
        split=str(raw.get("split", "test")), output_dir=Path(raw["output_dir"]), adapters=adapters,
        expected_records=raw.get("expected_records"), batch_size=int(raw.get("batch_size", 16)),
        max_new_tokens=int(raw.get("max_new_tokens", 1024)),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.85)),
        max_lora_rank=int(raw.get("max_lora_rank", 16)),
        session_limit_seconds=raw.get("session_limit_seconds"),
        previous_results_dir=Path(raw["previous_results_dir"]) if raw.get("previous_results_dir") else None,
        probe=bool(raw.get("probe", True)),
    )
    if not adapters or len({item.name for item in adapters}) != len(adapters):
        raise ValueError("Configure at least one adapter with a unique name")
    if any(not item.name or "/" in item.name or "\\" in item.name for item in adapters):
        raise ValueError("Adapter names must be nonempty directory names")
    if config.batch_size < 1 or config.max_new_tokens < 1 or config.max_lora_rank < 1:
        raise ValueError("Batch size, generation limit, and LoRA rank must be positive")
    if not 0 < config.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if config.session_limit_seconds is not None and config.session_limit_seconds < 1:
        raise ValueError("session_limit_seconds must be positive or null")
    if config.expected_records is not None and config.expected_records < 1:
        raise ValueError("expected_records must be positive or null")
    if not (config.model_path / "config.json").is_file():
        raise FileNotFoundError(config.model_path / "config.json")
    if not (config.dataset_root / f"{config.split}.jsonl").is_file():
        raise FileNotFoundError(config.dataset_root / f"{config.split}.jsonl")
    for adapter in adapters:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter.path / name).is_file():
                raise FileNotFoundError(adapter.path / name)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
