"""Validated configuration for LoRA training runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nemotron3_fc.paths import resolve_path

DEFAULT_LORA_TARGETS = ("up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    train: bool
    evaluate: bool
    expected_train_records: int | None
    expected_validation_records: int | None
    monitor_validation_records: int


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    expected_module_counts: Mapping[str, int] | None
    expected_trainable_parameters: int | None


@dataclass(frozen=True)
class TrainingConfig:
    model_path: Path
    output_dir: Path
    mode: str
    run_number: int
    start_from: str
    previous_checkpoint: Path | None
    previous_best: Path | None
    epochs: int
    seed: int
    monitor_seed: int
    max_tokens: int
    overlap_tokens: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    max_batch_examples: int
    max_batch_padded_tokens: int
    length_bucket_records: int
    quick_train_per_source: int
    quick_validation_per_source: int
    log_every: int
    validate_every: int
    checkpoint_every: int
    stop_after_session_seconds: int | None
    minimum_free_disk_gib: float
    checkpoint_minimum_free_disk_gib: float
    best_score_weights: Mapping[str, float]
    datasets: tuple[DatasetConfig, ...]
    lora: LoraConfig

    @property
    def train_datasets(self) -> tuple[DatasetConfig, ...]:
        return tuple(dataset for dataset in self.datasets if dataset.train)

    @property
    def evaluation_datasets(self) -> tuple[DatasetConfig, ...]:
        return tuple(dataset for dataset in self.datasets if dataset.evaluate)

    def identity(self) -> dict[str, Any]:
        """Return stable settings that must agree across checkpoint resumption."""
        # Epoch count is intentionally absent: a completed-epoch checkpoint may be
        # continued with more epochs while the per-update training contract stays
        # unchanged and is checked separately through the schedule hash.
        return {
            "model_path": str(self.model_path),
            "mode": self.mode,
            "run_number": self.run_number,
            "seed": self.seed,
            "monitor_seed": self.monitor_seed,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "max_batch_examples": self.max_batch_examples,
            "max_batch_padded_tokens": self.max_batch_padded_tokens,
            "length_bucket_records": self.length_bucket_records,
            "quick_train_per_source": self.quick_train_per_source,
            "quick_validation_per_source": self.quick_validation_per_source,
            "validate_every": self.validate_every,
            "checkpoint_every": self.checkpoint_every,
            "best_score_weights": dict(self.best_score_weights),
            "datasets": [
                {
                    "name": item.name,
                    "root": str(item.root),
                    "train": item.train,
                    "evaluate": item.evaluate,
                    "monitor_validation_records": item.monitor_validation_records,
                }
                for item in self.datasets
            ],
            "lora": _jsonable(asdict(self.lora)),
        }

    def identity_sha256(self) -> str:
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded = payload.encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_training_config(path: Path) -> TrainingConfig:
    """Load a JSON configuration and reject inconsistent training settings."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Training configuration root must be an object")

    base = path.parent
    datasets = _datasets(raw.get("datasets"), base)
    lora = _lora(raw.get("lora", {}))
    mode = str(raw.get("mode", "full"))
    start_from = str(raw.get("start_from", "fresh"))
    stop_seconds = raw.get("stop_after_session_seconds", None)

    # Parse scalar values before performing cross-field validation. This keeps
    # type errors distinct from inconsistent run-state errors.
    config = TrainingConfig(
        model_path=_path(base, raw.get("model_path"), "model_path"),
        output_dir=_path(base, raw.get("output_dir"), "output_dir"),
        mode=mode,
        run_number=_positive_int(raw.get("run_number", 1), "run_number"),
        start_from=start_from,
        previous_checkpoint=_optional_path(base, raw.get("previous_checkpoint")),
        previous_best=_optional_path(base, raw.get("previous_best")),
        epochs=_positive_int(raw.get("epochs", 1), "epochs"),
        seed=_integer(raw.get("seed", 2026), "seed"),
        monitor_seed=_integer(raw.get("monitor_seed", 3031), "monitor_seed"),
        max_tokens=_positive_int(raw.get("max_tokens", 4096), "max_tokens"),
        overlap_tokens=_nonnegative_int(raw.get("overlap_tokens", 256), "overlap_tokens"),
        learning_rate=_positive_number(raw.get("learning_rate", 1e-4), "learning_rate"),
        warmup_steps=_positive_int(raw.get("warmup_steps", 100), "warmup_steps"),
        weight_decay=_nonnegative_number(raw.get("weight_decay", 0.01), "weight_decay"),
        gradient_clip=_positive_number(raw.get("gradient_clip", 1.0), "gradient_clip"),
        max_batch_examples=_positive_int(raw.get("max_batch_examples", 4), "max_batch_examples"),
        max_batch_padded_tokens=_positive_int(raw.get("max_batch_padded_tokens", 3072), "max_batch_padded_tokens"),
        length_bucket_records=_positive_int(raw.get("length_bucket_records", 128), "length_bucket_records"),
        quick_train_per_source=_positive_int(raw.get("quick_train_per_source", 64), "quick_train_per_source"),
        quick_validation_per_source=_positive_int(
            raw.get("quick_validation_per_source", 32), "quick_validation_per_source"
        ),
        log_every=_positive_int(raw.get("log_every", 100), "log_every"),
        validate_every=_positive_int(raw.get("validate_every", 250), "validate_every"),
        checkpoint_every=_positive_int(raw.get("checkpoint_every", 250), "checkpoint_every"),
        stop_after_session_seconds=(
            None if stop_seconds is None else _positive_int(stop_seconds, "stop_after_session_seconds")
        ),
        minimum_free_disk_gib=_nonnegative_number(raw.get("minimum_free_disk_gib", 11), "minimum_free_disk_gib"),
        checkpoint_minimum_free_disk_gib=_nonnegative_number(
            raw.get("checkpoint_minimum_free_disk_gib", 6), "checkpoint_minimum_free_disk_gib"
        ),
        best_score_weights=_weights(raw.get("best_score_weights")),
        datasets=datasets,
        lora=lora,
    )
    _validate_config(config)
    return config


def _datasets(value: Any, base: Path) -> tuple[DatasetConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("datasets must be a non-empty list")

    result = []
    names = set()
    # Each configured dataset must participate in training, evaluation, or both.
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("Each dataset configuration must be an object")
        name = str(row.get("name", "")).strip().lower()
        if not name or name in names:
            raise ValueError(f"Dataset names must be non-empty and unique: {name!r}")
        names.add(name)

        train = bool(row.get("train", False))
        evaluate = bool(row.get("evaluate", False))
        if not train and not evaluate:
            raise ValueError(f"Dataset {name} is neither a training nor evaluation source")

        result.append(
            DatasetConfig(
                name=name,
                root=_path(base, row.get("root"), f"datasets[{name}].root"),
                train=train,
                evaluate=evaluate,
                expected_train_records=_optional_positive_int(
                    row.get("expected_train_records"), f"{name}.expected_train_records"
                ),
                expected_validation_records=_optional_positive_int(
                    row.get("expected_validation_records"), f"{name}.expected_validation_records"
                ),
                monitor_validation_records=(
                    _positive_int(row.get("monitor_validation_records", 200), f"{name}.monitor_validation_records")
                    if evaluate
                    else 0
                ),
            )
        )
    return tuple(result)


def _lora(value: Any) -> LoraConfig:
    if not isinstance(value, Mapping):
        raise ValueError("lora must be an object")

    targets = value.get("target_modules", list(DEFAULT_LORA_TARGETS))
    if not isinstance(targets, list) or not targets or any(not isinstance(item, str) or not item for item in targets):
        raise ValueError("lora.target_modules must be a non-empty string list")

    # Optional architecture expectations detect changes in LoRA module coverage.
    counts = value.get("expected_module_counts")
    if counts is not None:
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(targets)
            or any(not isinstance(item, int) or item < 0 for item in counts.values())
        ):
            raise ValueError("lora.expected_module_counts must define every target with non-negative counts")
        counts = {str(key): int(item) for key, item in counts.items()}

    trainable = _optional_positive_int(value.get("expected_trainable_parameters"), "lora.expected_trainable_parameters")
    dropout = _nonnegative_number(value.get("dropout", 0.0), "lora.dropout")
    if dropout >= 1:
        raise ValueError("lora.dropout must be less than 1")
    return LoraConfig(
        rank=_positive_int(value.get("rank", 16), "lora.rank"),
        alpha=_positive_int(value.get("alpha", 32), "lora.alpha"),
        dropout=dropout,
        target_modules=tuple(targets),
        expected_module_counts=counts,
        expected_trainable_parameters=trainable,
    )


def _validate_config(config: TrainingConfig) -> None:
    _validate_run_mode(config)
    _validate_token_windows(config)
    _validate_dataset_roles(config)
    _validate_selection_metrics(config)


def _validate_run_mode(config: TrainingConfig) -> None:
    """Validate fresh and resumed run state without touching checkpoint files."""
    if config.mode not in {"full", "quick-test"}:
        raise ValueError("mode must be 'full' or 'quick-test'")
    if config.start_from not in {"fresh", "checkpoint"}:
        raise ValueError("start_from must be 'fresh' or 'checkpoint'")

    has_previous_state = config.previous_checkpoint is not None or config.previous_best is not None
    has_complete_previous_state = config.previous_checkpoint is not None and config.previous_best is not None
    if config.start_from == "fresh" and has_previous_state:
        raise ValueError("Fresh runs must not specify previous_checkpoint or previous_best")
    if config.start_from == "checkpoint" and not has_complete_previous_state:
        raise ValueError("Checkpoint runs require previous_checkpoint and previous_best")


def _validate_token_windows(config: TrainingConfig) -> None:
    if config.overlap_tokens >= config.max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")


def _validate_dataset_roles(config: TrainingConfig) -> None:
    if not config.train_datasets or not config.evaluation_datasets:
        raise ValueError("At least one training and one evaluation dataset are required")


def _validate_selection_metrics(config: TrainingConfig) -> None:
    # Selection weights may target a complete dataset or a supported stratum.
    valid_metrics = {dataset.name for dataset in config.evaluation_datasets}
    valid_metrics.update(f"{dataset.name}:no_call" for dataset in config.evaluation_datasets)
    valid_metrics.update(f"{dataset.name}:multi_turn" for dataset in config.evaluation_datasets)
    unknown = set(config.best_score_weights) - valid_metrics
    if unknown:
        raise ValueError(f"best_score_weights contains unknown metrics: {sorted(unknown)}")


def _weights(value: Any) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("best_score_weights must be a non-empty object")
    result = {str(key): _positive_number(item, f"best_score_weights.{key}") for key, item in value.items()}
    return result


def _path(base: Path, value: Any, label: str) -> Path:
    return resolve_path(base, value, label)


def _optional_path(base: Path, value: Any) -> Path | None:
    return None if value is None else _path(base, value, "optional path")


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _optional_positive_int(value: Any, label: str) -> int | None:
    return None if value is None else _positive_int(value, label)


def _positive_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _nonnegative_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
