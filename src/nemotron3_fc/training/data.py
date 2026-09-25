"""Verified dataset loading, stratified validation, and deterministic batch schedules."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron3_fc.data.io import read_jsonl, sha256_file
from nemotron3_fc.data.schema import record_from_dict
from nemotron3_fc.training.config import DatasetConfig, TrainingConfig
from nemotron3_fc.training.encoding import EncodedWindow, record_windows


@dataclass(frozen=True)
class BatchItem:
    source: str
    id: str
    indices: tuple[int, ...]
    tokens: int
    supervised: int
    examples: int


@dataclass(frozen=True)
class PreparedData:
    training_windows: tuple[EncodedWindow, ...]
    batch_items: tuple[BatchItem, ...]
    schedule: tuple[tuple[int, int], ...]
    validation_monitor: tuple[EncodedWindow, ...]
    validation_full: tuple[EncodedWindow, ...]
    monitor_record_counts: Mapping[str, int]
    full_record_counts: Mapping[str, int]
    schedule_sha256: str
    counts: Mapping[str, Any]
    dataset_identities: Mapping[str, Any]


def load_verified_split(dataset: DatasetConfig, split: str) -> tuple[list[dict], dict]:
    """Verify a split against its manifest before returning validated canonical rows."""
    manifest_path = dataset.root / "manifest.json"
    data_path = dataset.root / f"{split}.jsonl"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing {dataset.name} {split} data or manifest under {dataset.root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != dataset.name:
        raise RuntimeError(f"Dataset manifest source mismatch for {dataset.name}")
    expected = manifest.get("files", {}).get(f"{split}.jsonl")
    if not isinstance(expected, Mapping) or expected.get("sha256") != sha256_file(data_path):
        raise RuntimeError(f"Dataset hash mismatch: {data_path}")
    rows = list(read_jsonl(data_path))
    if expected.get("records") != len(rows):
        raise RuntimeError(f"Dataset count mismatch: {data_path}")
    configured = dataset.expected_train_records if split == "train" else dataset.expected_validation_records
    if configured is not None and configured != len(rows):
        raise RuntimeError(f"Configured {dataset.name} {split} count is {configured}, manifest contains {len(rows)}")
    for row in rows:
        record = record_from_dict(row)
        if record.source != dataset.name:
            raise RuntimeError(f"Wrong source in {data_path}: {record.id}")
    return rows, {"records": len(rows), "sha256": expected["sha256"], "manifest": str(manifest_path)}


def prepare_data(config: TrainingConfig, tokenizer: Any) -> PreparedData:
    """Encode all configured records and construct an immutable multi-epoch schedule."""
    windows: list[EncodedWindow] = []
    groups: list[list[int]] = []
    counts: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    train_limit = config.quick_train_per_source if config.mode == "quick-test" else None

    # Encode complete records first so windows from one conversation stay grouped.
    for dataset in config.train_datasets:
        rows, identity = load_verified_split(dataset, "train")
        identities.setdefault(dataset.name, {})["train"] = identity
        if train_limit is not None:
            rows = rows[:train_limit]
        errors = []
        start_count = len(windows)
        for row in rows:
            try:
                encoded = record_windows(
                    row,
                    dataset.name,
                    tokenizer,
                    max_tokens=config.max_tokens,
                    overlap_tokens=config.overlap_tokens,
                )
                groups.append(list(range(len(windows), len(windows) + len(encoded))))
                windows.extend(encoded)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                errors.append({"source": dataset.name, "id": row.get("id"), "error": str(error)})
        print(
            f"TRAIN DATA source={dataset.name} records={len(rows)} "
            f"windows={len(windows) - start_count} errors={len(errors)}"
        )
        for error in errors[:5]:
            print(f"TRAIN DATA ERROR {json.dumps(error, ensure_ascii=False)}")
        if errors:
            raise RuntimeError(f"{dataset.name} contains {len(errors)} records that cannot be encoded")
        counts[f"{dataset.name}_train_records"] = len(rows)
        counts[f"{dataset.name}_train_windows"] = len(windows) - start_count

    batch_items: list[BatchItem] = []
    schedule: list[tuple[int, int]] = []
    updates_per_epoch = []

    # Build the deterministic optimizer update schedule for every configured epoch.
    for epoch in range(config.epochs):
        batches = make_epoch_batches(windows, groups, epoch, config)
        describe_batch_plan(windows, batches, epoch, config)
        updates_per_epoch.append(len(batches))

        for indices in batches:
            sources = {windows[index].source for index in indices}
            source = next(iter(sources)) if len(sources) == 1 else "mixed"

            batch_items.append(
                BatchItem(
                    source=source,
                    id="|".join(windows[index].id for index in indices),
                    indices=indices,
                    tokens=sum(windows[index].tokens for index in indices),
                    supervised=sum(windows[index].supervised for index in indices),
                    examples=len(indices),
                )
            )
            schedule.append((epoch, len(batch_items) - 1))

    monitor: list[EncodedWindow] = []
    full: list[EncodedWindow] = []
    monitor_counts, full_counts = {}, {}

    # Keep a stratified monitor for periodic checks and the complete split for epoch selection.
    for dataset_index, dataset in enumerate(config.evaluation_datasets):
        rows, identity = load_verified_split(dataset, "validation")
        identities.setdefault(dataset.name, {})["validation"] = identity

        if config.mode == "quick-test":
            rows = rows[: min(config.quick_validation_per_source, len(rows))]

        source_monitor, source_full, source_monitor_count = build_validation_sets(
            rows,
            dataset,
            tokenizer,
            config,
            seed=config.monitor_seed + dataset_index,
        )

        monitor.extend(source_monitor)
        full.extend(source_full)
        monitor_counts[dataset.name] = source_monitor_count
        full_counts[dataset.name] = len(rows)

    schedule_hash = schedule_sha256(schedule, batch_items)

    counts.update(
        {
            "windows_per_epoch": len(windows),
            "updates_per_epoch": updates_per_epoch,
            "monitor_validation_records": monitor_counts,
            "full_validation_records": full_counts,
        }
    )

    print(f"TOTAL epochs={config.epochs} windows={len(windows) * config.epochs} optimizer_updates={len(schedule)}")
    print(f"Training schedule SHA-256: {schedule_hash}")

    return PreparedData(
        training_windows=tuple(windows),
        batch_items=tuple(batch_items),
        schedule=tuple(schedule),
        validation_monitor=tuple(monitor),
        validation_full=tuple(full),
        monitor_record_counts=monitor_counts,
        full_record_counts=full_counts,
        schedule_sha256=schedule_hash,
        counts=counts,
        dataset_identities=identities,
    )


def build_validation_sets(
    rows: Sequence[Mapping[str, Any]],
    dataset: DatasetConfig,
    tokenizer: Any,
    config: TrainingConfig,
    *,
    seed: int,
) -> tuple[list[EncodedWindow], list[EncodedWindow], int]:
    """Encode full validation and select a reproducible proportional stratified monitor."""

    records = []
    seen_ids = set()
    errors = []

    for row in rows:
        record_id = str(row["id"])

        if record_id in seen_ids:
            raise RuntimeError(f"Duplicate validation record ID: {record_id}")
        seen_ids.add(record_id)

        try:
            encoded = record_windows(
                row,
                dataset.name,
                tokenizer,
                max_tokens=config.max_tokens,
                overlap_tokens=config.overlap_tokens,
            )
            first = encoded[0]
            records.append(
                {
                    "id": record_id,
                    "stratum": (first.no_call, first.multi_turn, first.has_tool_result),
                    "windows": encoded,
                }
            )
        except (ValueError, TypeError, KeyError, IndexError) as error:
            errors.append({"id": record_id, "error": str(error)})

    if errors:
        for error in errors[:5]:
            print(f"VALIDATION DATA ERROR {json.dumps(error, ensure_ascii=False)}")
        raise RuntimeError(f"{dataset.name} contains {len(errors)} validation records that cannot be encoded")
    if config.mode == "full" and dataset.monitor_validation_records > len(records):
        raise RuntimeError(
            f"{dataset.name} monitor requests {dataset.monitor_validation_records} of only {len(records)} records"
        )
    monitor_count = min(dataset.monitor_validation_records, len(records))
    if monitor_count < 1:
        raise RuntimeError(f"No validation records available for {dataset.name}")
    strata: dict[tuple[bool, bool, bool], list[int]] = {}
    for index, record in enumerate(records):
        strata.setdefault(record["stratum"], []).append(index)
    rng = random.Random(seed)
    for indices in strata.values():
        rng.shuffle(indices)
    if monitor_count < len(strata):
        raise RuntimeError(f"{dataset.name} monitor size {monitor_count} cannot represent all {len(strata)} strata")
    # Seed every stratum with one record, then allocate the remainder by
    # proportional deficit so small strata cannot disappear from monitoring.
    quotas = {key: 1 for key in strata}
    while sum(quotas.values()) < monitor_count:
        eligible = [key for key, indices in strata.items() if quotas[key] < len(indices)]
        if not eligible:
            raise RuntimeError(f"Cannot allocate {dataset.name} monitoring sample")
        key = max(
            eligible,
            key=lambda candidate: (
                monitor_count * len(strata[candidate]) / len(records) - quotas[candidate],
                len(strata[candidate]),
                candidate,
            ),
        )
        quotas[key] += 1
    chosen = {index for key, indices in strata.items() for index in indices[: quotas[key]]}
    monitor = [window for index, record in enumerate(records) if index in chosen for window in record["windows"]]
    full = [window for record in records for window in record["windows"]]
    print(
        f"VALIDATION source={dataset.name} full_records={len(records)} full_windows={len(full)} "
        f"monitor_records={len(chosen)} monitor_windows={len(monitor)}"
    )
    for key in sorted(strata):
        print(
            f"VALIDATION STRATUM source={dataset.name} no_call/multi_turn/tool_result={key} "
            f"full={len(strata[key])} monitor={quotas[key]}"
        )
    return monitor, full, len(chosen)


def make_epoch_batches(
    windows: Sequence[EncodedWindow], groups: Sequence[Sequence[int]], epoch: int, config: TrainingConfig
) -> list[tuple[int, ...]]:
    """Shuffle whole records, bucket lengths, and include every window exactly once."""
    rng = random.Random(config.seed + epoch)
    order = list(range(len(groups)))
    rng.shuffle(order)
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max = 0
    current_source: str | None = None
    for offset in range(0, len(order), config.length_bucket_records):
        block = order[offset : offset + config.length_bucket_records]
        block.sort(key=lambda group_id: max(windows[index].tokens for index in groups[group_id]))
        for group_id in block:
            for index in groups[group_id]:
                item = windows[index]
                proposed_max = max(current_max, item.tokens)
                proposed_count = len(current) + 1
                exceeds_budget = proposed_count > 1 and proposed_max * proposed_count > config.max_batch_padded_tokens
                # A batch-wide loss is token-weighted; keeping sources apart
                # preserves meaningful per-source metrics in mixed experiments.
                changes_source = current_source is not None and item.source != current_source
                if current and (changes_source or proposed_count > config.max_batch_examples or exceeds_budget):
                    batches.append(tuple(current))
                    current, current_max, current_source = [], 0, None
                current.append(index)
                current_max = max(current_max, item.tokens)
                current_source = item.source
    if current:
        batches.append(tuple(current))
    # Start the first epoch with a singleton as a low-memory smoke step before
    # the scheduler reaches ordinary packed batches.
    if epoch == 0 and batches and len(batches[0]) > 1:
        first = batches[0]
        batches[0:1] = [(first[0],), tuple(first[1:])]
    used = [index for batch in batches for index in batch]
    if len(used) != len(windows) or sorted(used) != list(range(len(windows))):
        raise RuntimeError(f"Epoch {epoch + 1} has duplicated or missing windows")
    return batches


def describe_batch_plan(
    windows: Sequence[EncodedWindow], batches: Sequence[Sequence[int]], epoch: int, config: TrainingConfig
) -> None:
    actual_tokens = sum(windows[index].tokens for batch in batches for index in batch)
    padded_tokens = sum(max(windows[index].tokens for index in batch) * len(batch) for batch in batches)
    sizes = [len(batch) for batch in batches]
    distribution = {size: sizes.count(size) for size in range(1, config.max_batch_examples + 1)}
    plan = {
        "epoch": epoch + 1,
        "windows": sum(sizes),
        "optimizer_updates": len(batches),
        "mean_examples_per_update": round(sum(sizes) / len(sizes), 2),
        "actual_tokens": actual_tokens,
        "padding_efficiency": round(actual_tokens / padded_tokens, 3),
        "batch_sizes": distribution,
    }
    print(f"EPOCH PLAN {json.dumps(plan)}")


def schedule_sha256(schedule: Sequence[tuple[int, int]], items: Sequence[BatchItem], count: int | None = None) -> str:
    # Hash semantic update order rather than Python object identities so the
    # schedule can be reconstructed and verified in another process.
    digest = hashlib.sha256()
    limit = len(schedule) if count is None else count
    for epoch, item_index in schedule[:limit]:
        item = items[item_index]
        digest.update(f"{epoch}|{item.source}|{item.id}|{item.tokens}|{item.supervised}\n".encode())
    return digest.hexdigest()


def schedule_identity(data: PreparedData, position: int) -> str:
    epoch, item_index = data.schedule[position]
    item = data.batch_items[item_index]
    return f"{epoch}|{item.source}|{item.id}"


def advance_progress_hash(current_hash: str, data: PreparedData, position: int) -> str:
    return hashlib.sha256((current_hash + "|" + schedule_identity(data, position)).encode("utf-8")).hexdigest()


def prefix_progress_hash(data: PreparedData, count: int) -> str:
    digest = hashlib.sha256(b"").hexdigest()
    for position in range(count):
        digest = advance_progress_hash(digest, data, position)
    return digest
