"""Summaries and paired comparisons over persisted per-turn predictions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nemotron3_fc.evaluation.scoring import BFCL_SOURCE_COMMIT
from nemotron3_fc.evaluation.tasks import read_jsonl

PRIMARY_EVALUATOR = "BFCL-v3-style AST and relevance/irrelevance applied to canonical held-out assistant turns"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def successful_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    # Task ID is the stable join key across resumed runs and adapter outputs.
    if path.is_file():
        for row in read_jsonl(path):
            if row.get("status") == "ok":
                if row["task_id"] in result:
                    raise RuntimeError(f"Duplicate successful prediction: {row['task_id']}")
                result[row["task_id"]] = row
    return result


def metric_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row["metrics"][field] for row in rows if row["metrics"].get(field) is not None]
    return sum(values) / len(values) if values else None


def summarize_adapter(
    name: str,
    step: int | None,
    successful: dict[str, dict[str, Any]],
    status: str,
    info: dict[str, Any],
    adapter_sha256: str,
    runtime: str,
) -> dict[str, Any]:
    # Call and no-call turns have different primary metrics and denominators.
    rows = list(successful.values())
    call_rows = [row for row in rows if row["metrics"]["expected_call"]]
    no_call_rows = [row for row in rows if not row["metrics"]["expected_call"]]
    seconds = sum(row["seconds"] for row in rows)
    tokens = sum(row["generated_tokens"] for row in rows)

    errors = Counter(
        row["metrics"].get("bfcl_ast_error_type") for row in call_rows if row["metrics"].get("bfcl_ast_error_type")
    )

    return {
        "adapter": name,
        "status": status,
        "adapter_step": step,
        "primary_evaluator": PRIMARY_EVALUATOR,
        "bfcl_source_commit": BFCL_SOURCE_COMMIT,
        "primary_bfcl_ast_accuracy": metric_mean(call_rows, "bfcl_ast_correct"),
        "primary_bfcl_relevance_accuracy": metric_mean(call_rows, "bfcl_relevance_correct"),
        "primary_bfcl_irrelevance_accuracy": metric_mean(no_call_rows, "bfcl_irrelevance_correct"),
        "primary_bfcl_decode_valid_rate": metric_mean(rows, "bfcl_decode_valid"),
        "bfcl_ast_error_counts": dict(errors),
        "expected_test_records": info["records"],
        "covered_test_records": len({row["record_id"] for row in rows}),
        "expected_assistant_turns": info["assistant_turns"],
        "successful_assistant_turns": len(rows),
        "call_turns_evaluated": len(call_rows),
        "no_call_turns_evaluated": len(no_call_rows),
        "secondary_custom_call_decision_accuracy": metric_mean(rows, "decision_correct"),
        "secondary_custom_no_call_decision_accuracy": metric_mean(no_call_rows, "decision_correct"),
        "secondary_custom_function_names_exact_rate": metric_mean(call_rows, "function_names_exact"),
        "secondary_custom_native_calls_exact_rate": metric_mean(call_rows, "native_calls_exact"),
        "secondary_custom_syntax_valid_rate": metric_mean(rows, "syntax_valid"),
        "secondary_custom_text_exact_rate": metric_mean(rows, "text_exact"),
        "outputs_at_token_limit": sum(row["finish_reason"] == "max_new_tokens" for row in rows),
        "generated_tokens": tokens,
        "generation_seconds": seconds,
        "generated_tokens_per_second": tokens / seconds if seconds else None,
        "adapter_sha256": adapter_sha256,
        "split_sha256": info["split_sha256"],
        "runtime": runtime,
    }


def paired_comparison(
    output_dir: Path, tasks: list[dict[str, Any]], first_name: str, second_name: str
) -> dict[str, Any]:
    """Compare exactly matching assistant turns from two adapter runs."""
    first = successful_predictions(output_dir / first_name / "predictions.jsonl")
    second = successful_predictions(output_dir / second_name / "predictions.jsonl")
    common = set(first) & set(second)
    fields = ["task_id", "record_id", "message_index", "expected_call"]
    fields += [
        f"{name}_{metric}"
        for name in (first_name, second_name)
        for metric in (
            "bfcl_ast_correct",
            "bfcl_relevance_correct",
            "bfcl_irrelevance_correct",
            "native_calls_exact",
            "finish_reason",
            "generated_tokens",
        )
    ]
    # Adapter outputs are aligned by stable task ID rather than file position.
    with (output_dir / "paired-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            key = task["task_id"]
            if key not in common:
                continue
            row = {
                "task_id": key,
                "record_id": task["record_id"],
                "message_index": task["message_index"],
                "expected_call": task["expected_call"],
            }
            for name, prediction in ((first_name, first[key]), (second_name, second[key])):
                for metric in (
                    "bfcl_ast_correct",
                    "bfcl_relevance_correct",
                    "bfcl_irrelevance_correct",
                    "native_calls_exact",
                ):
                    row[f"{name}_{metric}"] = prediction["metrics"][metric]
                row[f"{name}_finish_reason"] = prediction["finish_reason"]
                row[f"{name}_generated_tokens"] = prediction["generated_tokens"]
            writer.writerow(row)
    call_keys = [key for key in common if first[key]["metrics"]["expected_call"]]
    no_call_keys = [key for key in common if not first[key]["metrics"]["expected_call"]]

    summary = {
        "primary_evaluator": PRIMARY_EVALUATOR,
        "bfcl_source_commit": BFCL_SOURCE_COMMIT,
        "adapters": [first_name, second_name],
        "complete_for_both_adapters": len(common) == len(tasks),
        "paired_assistant_turns": len(common),
        "paired_call_turns": len(call_keys),
        "paired_no_call_turns": len(no_call_keys),
    }

    for name, rows in ((first_name, first), (second_name, second)):
        summary[f"{name}_primary_bfcl_ast_accuracy"] = metric_mean([rows[key] for key in call_keys], "bfcl_ast_correct")

        summary[f"{name}_primary_bfcl_relevance_accuracy"] = metric_mean(
            [rows[key] for key in call_keys], "bfcl_relevance_correct"
        )

        summary[f"{name}_primary_bfcl_irrelevance_accuracy"] = metric_mean(
            [rows[key] for key in no_call_keys], "bfcl_irrelevance_correct"
        )

        summary[f"{name}_secondary_custom_native_calls_exact_rate"] = metric_mean(
            [rows[key] for key in call_keys], "native_calls_exact"
        )

    summary[f"{first_name}_only_bfcl_ast_correct"] = sum(
        bool(first[key]["metrics"]["bfcl_ast_correct"]) and not bool(second[key]["metrics"]["bfcl_ast_correct"])
        for key in call_keys
    )

    summary[f"{second_name}_only_bfcl_ast_correct"] = sum(
        bool(second[key]["metrics"]["bfcl_ast_correct"]) and not bool(first[key]["metrics"]["bfcl_ast_correct"])
        for key in call_keys
    )

    summary["both_bfcl_ast_correct"] = sum(
        bool(first[key]["metrics"]["bfcl_ast_correct"]) and bool(second[key]["metrics"]["bfcl_ast_correct"])
        for key in call_keys
    )

    summary["both_bfcl_ast_incorrect"] = sum(
        not bool(first[key]["metrics"]["bfcl_ast_correct"]) and not bool(second[key]["metrics"]["bfcl_ast_correct"])
        for key in call_keys
    )

    write_json(output_dir / "comparison-summary.json", summary)
    return summary
