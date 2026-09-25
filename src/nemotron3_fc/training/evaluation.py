"""Teacher-forced validation and best-adapter selection."""

from __future__ import annotations

import gc
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron3_fc.data.io import sha256_file, write_json
from nemotron3_fc.training.checkpoint import capture_rng, restore_rng
from nemotron3_fc.training.config import TrainingConfig
from nemotron3_fc.training.data import PreparedData
from nemotron3_fc.training.model import gpu_batch, load_exact_adapter, restore_adapter, snapshot_adapter


@dataclass
class BestState:
    score: float = float("inf")
    step: int | None = None
    epoch: int | None = None
    kind: str | None = None
    snapshot: Mapping[str, Any] | None = None
    source: Path | None = None
    metrics: Mapping[str, Any] | None = None
    validation_scope: str | None = None


def evaluate_model(model: Any, data: PreparedData, config: TrainingConfig, position: int, scope: str) -> dict[str, Any]:
    """Measure the configured monitor or full held-out records without mixing scopes."""
    import torch

    if scope not in {"monitor", "full"}:
        raise ValueError(f"Unknown validation scope: {scope}")
    items = data.validation_monitor if scope == "monitor" else data.validation_full
    expected_counts = data.monitor_record_counts if scope == "monitor" else data.full_record_counts
    actual_counts = {
        source: len({item.record_id for item in items if item.source == source}) for source in expected_counts
    }
    if actual_counts != dict(expected_counts):
        raise RuntimeError(
            f"{scope} validation record counts differ: expected={dict(expected_counts)}, actual={actual_counts}"
        )

    # Validation must not consume dropout/random state that a resumed training
    # run would otherwise use for its next optimizer update.
    previous_mode, rng_state = model.training, capture_rng()
    model.eval()
    metric_names = []
    for dataset in config.evaluation_datasets:
        metric_names.extend((dataset.name, f"{dataset.name}:no_call", f"{dataset.name}:multi_turn"))
    totals = {name: [0.0, 0] for name in metric_names}
    try:
        with torch.inference_mode():
            for index, item in enumerate(items, start=1):
                value = float(model(**gpu_batch(item)).loss.item())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite {scope} loss at {item.id}")
                # Convert each window's mean token loss to a token sum so long
                # and short windows contribute by supervised tokens, not count.
                weighted = value * item.supervised
                for name, included in (
                    (item.source, True),
                    (f"{item.source}:no_call", item.no_call),
                    (f"{item.source}:multi_turn", item.multi_turn),
                ):
                    if included:
                        totals[name][0] += weighted
                        totals[name][1] += item.supervised
                if scope == "full" and (index % 250 == 0 or index == len(items)):
                    print(f"FULL VALIDATION PROGRESS windows={index}/{len(items)}")
    finally:
        restore_rng(rng_state)
        model.train(previous_mode)

    result: dict[str, Any] = {
        "step": position,
        "epoch_at_step": data.schedule[position - 1][0] + 1 if position else 0,
        "scope": scope,
        "records": actual_counts,
        "windows": len(items),
        "warnings": 0,
        "validation_tokens": {name: tokens for name, (_, tokens) in totals.items()},
    }
    result.update({name: weighted / tokens if tokens else None for name, (weighted, tokens) in totals.items()})
    # Renormalize when an optional subgroup has no supervised tokens; otherwise
    # an unavailable metric would artificially lower the selection score.
    available = {name: result[name] for name in config.best_score_weights if result.get(name) is not None}
    denominator = sum(config.best_score_weights[name] for name in available)
    result["score"] = (
        sum(config.best_score_weights[name] * value for name, value in available.items()) / denominator
        if denominator
        else None
    )
    if result["score"] is None:
        raise RuntimeError(f"{scope} validation produced no selection score")
    visible = {key: value for key, value in result.items() if key != "validation_tokens"}
    print(f"VALIDATION {json.dumps(visible, ensure_ascii=False)}")
    return result


def previous_best_state(config: TrainingConfig) -> BestState:
    """Read previous best metadata without trusting its validation scope."""
    if config.previous_best is None:
        return BestState()
    root = config.previous_best
    metadata = json.loads((root / "best-metadata.json").read_text(encoding="utf-8"))
    if not (root / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Previous best adapter weights are missing: {root}")
    if metadata.get("run_number") != config.run_number:
        raise RuntimeError("Previous best adapter belongs to another run number")
    if not isinstance(metadata.get("score"), (int, float)) or not isinstance(metadata.get("global_step"), int):
        raise RuntimeError("Previous best metadata lacks a numeric score or global step")
    result = BestState(
        score=float(metadata["score"]),
        step=metadata.get("global_step"),
        epoch=metadata.get("completed_epoch"),
        kind=metadata.get("selection_kind", "provisional"),
        source=root,
        metrics=metadata.get("metrics"),
        validation_scope=metadata.get("validation_scope", "legacy"),
    )
    print(f"PREVIOUS BEST kind={result.kind} epoch={result.epoch} score={result.score} scope={result.validation_scope}")
    return result


def revalidate_previous_best(
    model: Any, best: BestState, data: PreparedData, config: TrainingConfig
) -> tuple[BestState, dict[str, Any] | None]:
    """Evaluate a legacy/non-full best on the current full split, then restore latest weights."""
    # Legacy or provisional selections are not comparable to full-split epoch
    # scores until they are measured on the same validation scope.
    if best.source is None or best.validation_scope == "full":
        return best, None
    latest = snapshot_adapter(model)
    try:
        load_exact_adapter(model, best.source / "adapter_model.safetensors")
        validation = evaluate_model(model, data, config, int(best.step), "full")
    finally:
        restore_adapter(model, latest)
        del latest
        gc.collect()
    best.score = validation["score"]
    best.validation_scope = "full"
    best.metrics = {"validation": validation, "revalidated_previous_best": True}
    print(f"PREVIOUS BEST REVALIDATED epoch={best.epoch} step={best.step} full_score={best.score}")
    return best, validation


def consider_best(
    model: Any,
    best: BestState,
    validation: Mapping[str, Any],
    *,
    completed_epoch: int | None = None,
    epoch_metrics: Mapping[str, Any] | None = None,
) -> BestState:
    """Select completed epochs using full validation; monitor results remain provisional."""
    score, step = float(validation["score"]), int(validation["step"])
    if step == 0:
        return best
    # Once any complete epoch exists, periodic monitor improvements may not
    # displace it because monitor and full-split scores have different support.
    if completed_epoch is not None:
        if validation["scope"] != "full":
            raise RuntimeError("A completed epoch requires full validation")
        should_replace = best.kind != "completed_epoch" or score < best.score - 1e-5
    else:
        if validation["scope"] != "monitor":
            raise RuntimeError("Periodic validation must use the monitor")
        if best.kind == "completed_epoch":
            return best
        should_replace = best.kind is None or score < best.score - 1e-5
    if not should_replace:
        return best
    result = BestState(
        score=score,
        step=step,
        epoch=completed_epoch,
        kind="completed_epoch" if completed_epoch is not None else "provisional",
        snapshot=snapshot_adapter(model),
        metrics=epoch_metrics if completed_epoch is not None else {"validation": dict(validation)},
        validation_scope=str(validation["scope"]),
    )
    print(
        f"NEW BEST kind={result.kind} epoch={result.epoch} step={result.step} "
        f"score={result.score:.6f} scope={result.validation_scope}"
    )
    gc.collect()
    return result


def export_best_adapter(model: Any, best: BestState, config: TrainingConfig, data: PreparedData) -> Path | None:
    """Export selected weights and their exact selection metadata."""
    if best.kind is None:
        print("BEST ADAPTER unavailable: no trained validation result")
        return None
    output = config.output_dir / "best-adapter"
    if output.exists():
        raise FileExistsError(f"Best-adapter output already exists: {output}")
    # Temporarily swap only LoRA tensors; the large base model remains resident
    # and unchanged throughout export.
    if best.snapshot is not None:
        latest = snapshot_adapter(model)
        try:
            restore_adapter(model, best.snapshot)
            model.save_pretrained(output, selected_adapters=["default"], safe_serialization=True)
        finally:
            restore_adapter(model, latest)
    elif best.source is not None:
        shutil.copytree(best.source, output)
        print("BEST ADAPTER previous run remains best")
    else:
        raise RuntimeError("Best adapter has neither saved weights nor an in-memory snapshot")
    metadata = {
        "score": best.score,
        "global_step": best.step,
        "completed_epoch": best.epoch,
        "selection_kind": best.kind,
        "validation_scope": best.validation_scope,
        "metric": "weighted_teacher_forced_validation_loss",
        "run_number": config.run_number,
        "schedule_sha256": data.schedule_sha256,
        "metrics": best.metrics,
    }
    write_json(output / "best-metadata.json", metadata)
    weights = output / "adapter_model.safetensors"
    print(
        f"BEST ADAPTER kind={best.kind} epoch={best.epoch} scope={best.validation_scope} "
        f"score={best.score} SHA-256={sha256_file(weights)}"
    )
    return output
