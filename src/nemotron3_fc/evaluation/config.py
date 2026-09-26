"""Evaluation configuration loading and input validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron3_fc.paths import resolve_path


@dataclass(frozen=True)
class AdapterConfig:
    """One LoRA adapter included in an evaluation run."""

    name: str
    path: Path
    step: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated inputs and runtime limits for turn-level evaluation."""

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
    """Load the JSON file and validate it before model initialization."""
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Evaluation configuration root must be an object")

    config = _build_config(raw, path.parent)
    _validate_settings(config)
    _validate_input_files(config)
    return config


def _build_config(raw: Mapping[str, Any], base: Path) -> EvaluationConfig:
    """Convert untrusted JSON values into the typed configuration model."""
    previous_results = raw.get("previous_results_dir")

    return EvaluationConfig(
        model_path=resolve_path(base, raw["model_path"], "model_path"),
        dataset_root=resolve_path(base, raw["dataset_root"], "dataset_root"),
        split=str(raw.get("split", "test")),
        output_dir=resolve_path(base, raw["output_dir"], "output_dir"),
        adapters=_parse_adapters(raw.get("adapters"), base),
        expected_records=_optional_positive_int(raw.get("expected_records"), "expected_records"),
        batch_size=_positive_int(raw.get("batch_size", 16), "batch_size"),
        max_new_tokens=_positive_int(raw.get("max_new_tokens", 1024), "max_new_tokens"),
        gpu_memory_utilization=_fraction(raw.get("gpu_memory_utilization", 0.85)),
        max_lora_rank=_positive_int(raw.get("max_lora_rank", 16), "max_lora_rank"),
        session_limit_seconds=_optional_positive_int(raw.get("session_limit_seconds"), "session_limit_seconds"),
        previous_results_dir=(
            resolve_path(base, previous_results, "previous_results_dir") if previous_results else None
        ),
        probe=bool(raw.get("probe", True)),
    )


def _parse_adapters(value: Any, base: Path) -> tuple[AdapterConfig, ...]:
    """Parse adapters separately so config construction stays readable."""
    if not isinstance(value, list) or not value:
        raise ValueError("Configure at least one adapter")

    # Preserve declaration order because it controls stable vLLM LoRA IDs and
    # paired-comparison ordering.
    adapters = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Each adapter configuration must be an object")

        name = str(item.get("name", "")).strip()
        adapters.append(
            AdapterConfig(
                name=name,
                path=resolve_path(base, item.get("path"), f"adapter {name!r} path"),
                step=_optional_nonnegative_int(item.get("step"), f"adapter {name!r} step"),
            )
        )

    return tuple(adapters)


def _validate_settings(config: EvaluationConfig) -> None:
    """Validate relationships that cannot be expressed by individual parsers."""
    names = [adapter.name for adapter in config.adapters]
    if len(set(names)) != len(names):
        raise ValueError("Adapter names must be unique")
    if any(not name or "/" in name or "\\" in name for name in names):
        raise ValueError("Adapter names must be nonempty directory names")
    if not config.split.strip() or "/" in config.split or "\\" in config.split:
        raise ValueError("split must be a nonempty filename component")


def _validate_input_files(config: EvaluationConfig) -> None:
    """Check every immutable input before allocating GPU memory."""
    required_files = [
        config.model_path / "config.json",
        config.dataset_root / f"{config.split}.jsonl",
    ]
    for adapter in config.adapters:
        required_files.extend(
            (
                adapter.path / "adapter_config.json",
                adapter.path / "adapter_model.safetensors",
            )
        )

    # Report every missing immutable input in one startup failure, before vLLM
    # allocates GPU memory.
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Evaluation inputs are missing:\n{formatted}")


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    return None if value is None else _positive_int(value, label)


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer or null")
    return value


def _fraction(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("gpu_memory_utilization must be numeric")

    result = float(value)
    if not 0 < result < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    return result


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
