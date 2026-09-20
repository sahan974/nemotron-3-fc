"""End-to-end resumable LoRA training session."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from nemotron3_fc.data.io import read_jsonl, write_json
from nemotron3_fc.training.checkpoint import initial_progress_hash, load_checkpoint, save_checkpoint
from nemotron3_fc.training.config import TrainingConfig
from nemotron3_fc.training.data import PreparedData, advance_progress_hash, prepare_data
from nemotron3_fc.training.evaluation import BestState, consider_best, evaluate_model, export_best_adapter, previous_best_state, revalidate_previous_best
from nemotron3_fc.training.model import build_training_model, load_tokenizer, new_optimizer_scheduler, train_one_step, verify_batch_loss_parity
from nemotron3_fc.training.report import create_training_report


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Validate inputs, build the deterministic schedule, and execute one training session."""
    session_started = time.monotonic()
    _verify_runtime_and_inputs(config)
    tokenizer = load_tokenizer(config)
    data = prepare_data(config, tokenizer)
    model = build_training_model(config)
    import torch

    print(f"GPU allocated after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(pad_id, int):
        raise RuntimeError("Tokenizer has no usable padding token")
    verify_batch_loss_parity(model, data.training_windows, pad_id)
    result = _run_session(model, tokenizer, data, config, session_started)
    try:
        report = create_training_report(config, result)
        result["training_report"] = str(config.output_dir / "training-report.json")
        result["training_plot"] = report["plot"]
        write_json(config.output_dir / "summary.json", result)
    except Exception as error:
        print(f"REPORT FAILED error={error!r}; saved training outputs remain available")
    return result


def _verify_runtime_and_inputs(config: TrainingConfig) -> None:
    try:
        import peft
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("Install the training dependencies before running train") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")
    if not (config.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model config is missing: {config.model_path}")
    if config.output_dir.exists():
        raise FileExistsError(f"Training output already exists: {config.output_dir}")
    if config.start_from == "checkpoint":
        checkpoint = config.previous_checkpoint
        best = config.previous_best
        required = (
            checkpoint / "manifest.json",
            checkpoint / "state.json",
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
            checkpoint / "adapter" / "adapter_model.safetensors",
            best / "best-metadata.json",
            best / "adapter_model.safetensors",
            checkpoint.parents[1] / "metrics.jsonl",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Resume artifacts are incomplete: {missing}")
    print(f"RUNTIME torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
    print(f"GPU name={torch.cuda.get_device_name(0)} vram_gib={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    print(f"TRAINING mode={config.mode} epochs={config.epochs} warmup_steps={config.warmup_steps} constant_lr={config.learning_rate}")
    print(f"OUTPUT {config.output_dir}")


def _run_session(model: Any, tokenizer: Any, data: PreparedData, config: TrainingConfig, session_started: float) -> dict[str, Any]:
    import torch

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(config.output_dir.parent).free / 1024**3
    if free_gib < config.minimum_free_disk_gib:
        raise OSError(f"Only {free_gib:.2f} GiB free before training; checkpoint replacement is unsafe")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(config.output_dir / "run-manifest.json", {
        "schema_version": 3,
        "mode": config.mode,
        "run_number": config.run_number,
        "start_from": config.start_from,
        "epochs": config.epochs,
        "total_updates": len(data.schedule),
        "data_counts": data.counts,
        "dataset_identities": data.dataset_identities,
        "schedule_sha256": data.schedule_sha256,
        "training_identity_sha256": config.identity_sha256(),
        "model_path": str(config.model_path),
        "torch": torch.__version__,
        "config": config.identity(),
    })

    best = previous_best_state(config)
    optimizer, scheduler = new_optimizer_scheduler(model, config)
    position, tokens_processed = 0, 0
    progress_hash = initial_progress_hash()
    if config.start_from == "checkpoint":
        state, optimizer, scheduler = load_checkpoint(model, config, data)
        position = state["next_position"]
        tokens_processed = state["tokens_processed"]
        progress_hash = state["progress_hash"]
        _carry_forward_metrics(config.previous_checkpoint, config.output_dir, position)

    metrics_file = config.output_dir / "metrics.jsonl"
    validation_file = config.output_dir / "validation.jsonl"
    epoch_file = config.output_dir / "epoch-summary.jsonl"
    recent: list[dict[str, Any]] = []
    checkpoint: Path | None = None
    status = "running"
    torch.cuda.reset_peak_memory_stats()
    print(f"SESSION start_step={position} total={len(data.schedule)} elapsed_minutes={(time.monotonic() - session_started) / 60:.1f}")

    best, prior_full = revalidate_previous_best(model, best, data, config)
    if prior_full is not None:
        _append_jsonl(validation_file, prior_full)
        _record_revalidated_prior_epoch(epoch_file, best, prior_full)
    initial_validation = evaluate_model(model, data, config, position, "monitor")
    _append_jsonl(validation_file, initial_validation)
    last_validation_step = position

    for index in range(position, len(data.schedule)):
        if config.stop_after_session_seconds is not None and time.monotonic() - session_started >= config.stop_after_session_seconds:
            status = "session_time_limit"
            print(f"SESSION STOP safe_step_boundary={index}")
            break
        epoch_index, item_index = data.schedule[index]
        item = data.batch_items[item_index]
        try:
            trained = train_one_step(model, data.training_windows, item.indices, _pad_id(tokenizer), item.supervised, optimizer, scheduler, config.gradient_clip)
        except Exception as error:
            status = "training_error"
            print(f"TRAINING STOP step={index} error={error!r}")
            print("The previous complete checkpoint, if any, remains recoverable.")
            break
        result = {
            "position": index,
            "epoch": epoch_index + 1,
            "source": item.source,
            "id": item.id,
            "tokens": item.tokens,
            "supervised": item.supervised,
            "examples": item.examples,
            **trained,
        }
        position = index + 1
        tokens_processed += item.tokens
        progress_hash = advance_progress_hash(progress_hash, data, index)
        recent.append(result)
        _append_jsonl(metrics_file, result)
        if len(recent) >= config.log_every:
            _print_training_window(recent, position, len(data.schedule))
            recent.clear()

        # Full validation is an epoch-boundary operation; periodic monitoring is
        # used only between completed epochs.
        epoch_end = position == len(data.schedule) or data.schedule[position][0] != data.schedule[position - 1][0]
        periodic_validation = position % config.validate_every == 0
        if periodic_validation:
            monitor = evaluate_model(model, data, config, position, "monitor")
            _append_jsonl(validation_file, monitor)
            last_validation_step = position
            if not epoch_end:
                best = consider_best(model, best, monitor)
        if epoch_end:
            full = evaluate_model(model, data, config, position, "full")
            _append_jsonl(validation_file, full)
            last_validation_step = position
            completed_epoch = data.schedule[position - 1][0] + 1
            epoch_metrics = _summarize_completed_epoch(completed_epoch, metrics_file, full, data)
            _append_jsonl(epoch_file, epoch_metrics)
            best = consider_best(model, best, full, completed_epoch=completed_epoch, epoch_metrics=epoch_metrics)
        if epoch_end or position % config.checkpoint_every == 0:
            try:
                checkpoint = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, config=config, data=data, position=position, tokens_processed=tokens_processed, progress_hash=progress_hash, best_score=best.score)
            except Exception as error:
                status = "checkpoint_error"
                print(f"CHECKPOINT STOP error={error!r}; previous complete checkpoint retained")
                break
    else:
        status = "completed"

    if recent:
        _print_training_window(recent, position, len(data.schedule))
    if status not in {"training_error", "checkpoint_error"} and position > 0:
        if last_validation_step != position:
            monitor = evaluate_model(model, data, config, position, "monitor")
            _append_jsonl(validation_file, monitor)
            best = consider_best(model, best, monitor)
        if checkpoint is None or checkpoint.name != f"checkpoint-step-{position:06d}":
            try:
                checkpoint = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, config=config, data=data, position=position, tokens_processed=tokens_processed, progress_hash=progress_hash, best_score=best.score)
            except Exception as error:
                status = "checkpoint_error"
                print(f"FINAL CHECKPOINT FAILED error={error!r}")

    best_output = None
    if checkpoint is not None and status != "checkpoint_error":
        try:
            best_output = export_best_adapter(model, best, config, data)
        except Exception as error:
            status = "best_export_error"
            print(f"BEST EXPORT FAILED error={error!r}; checkpoint remains available")
    summary = {
        "status": status,
        "mode": config.mode,
        "run_number": config.run_number,
        "next_position": position,
        "total_updates": len(data.schedule),
        "tokens_processed": tokens_processed,
        "session_minutes": (time.monotonic() - session_started) / 60,
        "latest_checkpoint": str(checkpoint) if checkpoint else None,
        "best_adapter": str(best_output) if best_output else None,
        "best_selection_kind": best.kind,
        "best_validation_scope": best.validation_scope,
        "best_completed_epoch": best.epoch,
        "best_step": best.step,
        "best_score": best.score if math.isfinite(best.score) else None,
        "best_metrics": best.metrics,
    }
    write_json(config.output_dir / "summary.json", summary)
    visible = {key: value for key, value in summary.items() if key != "best_metrics"}
    print(f"RUN SUMMARY {json.dumps(visible, indent=2)}")
    return summary


def _pad_id(tokenizer: Any) -> int:
    result = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(result, int):
        raise RuntimeError("Tokenizer has no usable padding token")
    return result


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _carry_forward_metrics(previous_checkpoint: Path, output_dir: Path, position: int) -> None:
    """Copy only measurements covered by the attached complete checkpoint."""
    previous_run = previous_checkpoint.parents[1]
    rules: tuple[tuple[str, Callable[[Mapping[str, Any]], bool]], ...] = (
        ("metrics.jsonl", lambda row: row["position"] < position),
        ("validation.jsonl", lambda row: row["step"] <= position),
        ("epoch-summary.jsonl", lambda row: row["end_step"] <= position),
    )
    for filename, keep in rules:
        source, target = previous_run / filename, output_dir / filename
        if not source.is_file():
            if filename == "metrics.jsonl":
                raise FileNotFoundError(f"Cannot reconstruct epoch metrics without {source}")
            continue
        kept = 0
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in read_jsonl(source):
                if keep(row):
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    kept += 1
        print(f"HISTORY file={filename} carried_rows={kept}")
    carried = list(read_jsonl(output_dir / "metrics.jsonl"))
    if len(carried) != position or any(row["position"] != index for index, row in enumerate(carried)):
        raise RuntimeError(f"Training metric history does not match checkpoint position {position}")


def _print_training_window(records: list[Mapping[str, Any]], position: int, total_updates: int) -> None:
    seconds = sum(row["seconds"] for row in records)
    tokens = sum(row["tokens"] for row in records)
    report = {
        "step": position,
        "of": total_updates,
        "epoch": records[-1]["epoch"],
        "window_steps": len(records),
        "mean_loss": round(sum(row["loss"] for row in records) / len(records), 5),
        "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
        "mean_step_seconds": round(seconds / len(records), 2),
        "max_grad_norm": round(max(row["grad_norm"] for row in records), 4),
        "lr": records[-1]["lr"],
    }
    try:
        import torch
        report["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except ImportError:
        report["peak_gpu_gib"] = None
    print(f"TRAIN {json.dumps(report)}")


def _summarize_completed_epoch(epoch: int, metrics_file: Path, validation: Mapping[str, Any], data: PreparedData) -> dict[str, Any]:
    """Aggregate every update in an epoch and attach its full validation result."""
    import torch

    if validation["scope"] != "full":
        raise RuntimeError("Epoch summary requires full validation")
    rows = [row for row in read_jsonl(metrics_file) if row["epoch"] == epoch]
    expected = sum(scheduled_epoch == epoch - 1 for scheduled_epoch, _ in data.schedule)
    if len(rows) != expected:
        raise RuntimeError(f"Epoch {epoch} has {len(rows)}/{expected} recorded updates")

    def source_loss(source: str | None = None) -> float | None:
        selected = [row for row in rows if source is None or row["source"] == source]
        tokens = sum(row["supervised"] for row in selected)
        return sum(row["loss"] * row["supervised"] for row in selected) / tokens if tokens else None

    seconds = sum(row["seconds"] for row in rows)
    sources = sorted({row["source"] for row in rows})
    result = {
        "epoch": epoch,
        "end_step": validation["step"],
        "updates": len(rows),
        "updates_by_source": {source: sum(row["source"] == source for row in rows) for source in sources},
        "training_windows": sum(row["examples"] for row in rows),
        "train_loss": source_loss(),
        "train_loss_by_source": {source: source_loss(source) for source in sources},
        "training_tokens": sum(row["tokens"] for row in rows),
        "supervised_tokens": sum(row["supervised"] for row in rows),
        "training_seconds": seconds,
        "tokens_per_second": sum(row["tokens"] for row in rows) / seconds if seconds else None,
        "mean_grad_norm": sum(row["grad_norm"] for row in rows) / len(rows),
        "max_grad_norm": max(row["grad_norm"] for row in rows),
        "lr_first": rows[0]["lr"],
        "lr_last": rows[-1]["lr"],
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "validation_scope": "full",
        "validation": dict(validation),
        "selection_score": validation["score"],
    }
    print(f"EPOCH COMPLETE {json.dumps({'epoch': epoch, 'windows': result['training_windows'], 'updates': result['updates'], 'train_loss': result['train_loss'], 'full_validation_score': result['selection_score']})}")
    return result


def _record_revalidated_prior_epoch(epoch_file: Path, best: BestState, validation: Mapping[str, Any]) -> None:
    if best.kind != "completed_epoch" or best.epoch is None or best.step is None:
        return
    old_rows = list(read_jsonl(epoch_file)) if epoch_file.is_file() else []
    matches = [row for row in old_rows if row.get("epoch") == best.epoch and row.get("end_step") == best.step]
    if not matches:
        raise RuntimeError("Cannot find the previous best epoch's recorded training metrics")
    row = dict(matches[-1])
    row["validation_scope"] = "full"
    row["validation"] = dict(validation)
    row["selection_score"] = validation["score"]
    row["revalidated_previous_best"] = True
    _append_jsonl(epoch_file, row)
    best.metrics = row
    print(f"PRIOR EPOCH FULL RESULT epoch={row['epoch']} step={row['end_step']} score={row['selection_score']}")
