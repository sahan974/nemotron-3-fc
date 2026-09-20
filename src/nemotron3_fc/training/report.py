"""Persistent metrics summary and separate-scale training plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nemotron3_fc.data.io import read_jsonl, write_json
from nemotron3_fc.training.config import TrainingConfig


def smoothed(values: Sequence[float], window: int) -> list[float]:
    """Compute a trailing display-only mean without changing stored measurements."""
    result, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def create_training_report(config: TrainingConfig, run_result: Mapping[str, Any]) -> dict[str, Any]:
    """Rank only full-validation epochs and render each metric on its own scale."""
    import matplotlib.pyplot as plt

    root = config.output_dir
    steps = list(read_jsonl(root / "metrics.jsonl")) if (root / "metrics.jsonl").is_file() else []
    validations = list(read_jsonl(root / "validation.jsonl")) if (root / "validation.jsonl").is_file() else []
    epochs = list(read_jsonl(root / "epoch-summary.jsonl")) if (root / "epoch-summary.jsonl").is_file() else []
    full_epochs = [row for row in epochs if row.get("validation_scope") == "full" and row.get("selection_score") is not None]
    best_epoch = min(full_epochs, key=lambda row: row["selection_score"]) if full_epochs else None
    write_json(root / "best-epoch-summary.json", {
        "selection_metric": "weighted_teacher_forced_validation_loss",
        "validation_scope": "full",
        "best_completed_epoch": best_epoch["epoch"] if best_epoch else None,
        "metrics": best_epoch,
        "note": None if best_epoch else "No completed epoch has been evaluated on the full split yet.",
    })

    # Give each validation metric its own axis so aggregate loss does not flatten
    # the smaller no-call or multi-turn curves.
    validation_metrics = []
    for dataset in config.evaluation_datasets:
        validation_metrics.extend((dataset.name, f"{dataset.name}:no_call", f"{dataset.name}:multi_turn"))
    validation_metrics.append("score")
    panel_names = ["training_loss", *validation_metrics, "learning_rate", "gradient_norm", "throughput", "gpu_memory"]
    columns = 3
    rows_count = (len(panel_names) + columns - 1) // columns
    figure, axes_grid = plt.subplots(rows_count, columns, figsize=(17, 4.2 * rows_count), constrained_layout=True)
    axes = list(getattr(axes_grid, "flat", [axes_grid]))
    panels = dict(zip(panel_names, axes))
    if steps:
        x = [row["position"] + 1 for row in steps]
        window = min(200, max(5, len(steps) // 100))
        panels["training_loss"].plot(x, smoothed([row["loss"] for row in steps], window), linewidth=1.2)
        panels["learning_rate"].plot(x, [row["lr"] for row in steps], linewidth=1.2)
        panels["gradient_norm"].plot(x, smoothed([row["grad_norm"] for row in steps], window), linewidth=1.2)
        panels["throughput"].plot(x, smoothed([row["tokens"] / row["seconds"] if row["seconds"] else 0 for row in steps], window), linewidth=1.2)
    monitor_rows = [row for row in validations if row.get("scope") == "monitor"]
    full_rows = [row for row in validations if row.get("scope") == "full"]
    for metric in validation_metrics:
        monitor_points = [(row["step"], row[metric]) for row in monitor_rows if row.get(metric) is not None]
        full_points = [(row["step"], row[metric]) for row in full_rows if row.get(metric) is not None]
        axis = panels[metric]
        if monitor_points:
            axis.plot([step for step, _ in monitor_points], [value for _, value in monitor_points], marker=".", markersize=3, linewidth=1.0, label="stratified monitor")
        if full_points:
            axis.scatter([step for step, _ in full_points], [value for _, value in full_points], marker="o", s=32, label="full validation")
        if monitor_points or full_points:
            axis.legend(fontsize=7)
    if epochs:
        panels["gpu_memory"].plot([row["epoch"] for row in epochs], [row["peak_gpu_gib"] for row in epochs], marker="o", linewidth=1.2)
    if best_epoch is not None:
        for metric in validation_metrics:
            panels[metric].axvline(best_epoch["end_step"], color="black", linestyle="--", linewidth=1, alpha=0.5)
    titles = {
        "training_loss": "Training loss",
        "score": "Validation selection score",
        "learning_rate": "Learning rate",
        "gradient_norm": "Gradient norm",
        "throughput": "Training throughput",
        "gpu_memory": "GPU memory",
    }
    for dataset in config.evaluation_datasets:
        titles[dataset.name] = f"{dataset.name} validation loss"
        titles[f"{dataset.name}:no_call"] = f"{dataset.name} no-call validation loss"
        titles[f"{dataset.name}:multi_turn"] = f"{dataset.name} multi-turn validation loss"
    for name, axis in panels.items():
        axis.set(title=titles[name], xlabel="Completed epoch" if name == "gpu_memory" else "Optimizer step")
        axis.grid(alpha=0.25)
    for axis in axes[len(panel_names) :]:
        axis.set_visible(False)
    plot_file = root / "training-curves.png"
    figure.savefig(plot_file, dpi=125, bbox_inches="tight")
    plt.close(figure)

    report = {
        "run_number": config.run_number,
        "mode": config.mode,
        "learning_rate_policy": "linear_warmup_then_constant",
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "recorded_training_updates": len(steps),
        "monitor_checks": len(monitor_rows),
        "full_checks": len(full_rows),
        "completed_epochs": len(epochs),
        "full_validated_epochs": len(full_epochs),
        "best_completed_epoch": best_epoch["epoch"] if best_epoch else None,
        "best_completed_epoch_metrics": best_epoch,
        "selected_adapter_kind": run_result.get("best_selection_kind"),
        "selected_adapter_validation_scope": run_result.get("best_validation_scope"),
        "selected_adapter_path": run_result.get("best_adapter"),
        "latest_checkpoint": run_result.get("latest_checkpoint"),
        "metrics_jsonl": str(root / "metrics.jsonl"),
        "validation_jsonl": str(root / "validation.jsonl"),
        "epoch_summary_jsonl": str(root / "epoch-summary.jsonl"),
        "plot": str(plot_file),
    }
    write_json(root / "training-report.json", report)
    print(f"REPORT path={root / 'training-report.json'} best_full_validated_epoch={report['best_completed_epoch']}")
    return report
