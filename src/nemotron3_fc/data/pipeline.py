"""Config-driven source conversion, leakage-safe splitting, and verified output writing."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron3_fc.data.io import load_raw_records, read_jsonl, sha256_file, write_json, write_jsonl
from nemotron3_fc.data.leakage import audit_split_leakage
from nemotron3_fc.data.registry import DatasetRegistry
from nemotron3_fc.data.schema import CanonicalRecord, record_from_dict
from nemotron3_fc.data.splitting import SPLIT_NAMES, plan_splits
from nemotron3_fc.paths import resolve_path

CONVERSION_ERRORS = (ValueError, TypeError, KeyError, IndexError, RecursionError)


@dataclass(frozen=True)
class SourceConfig:
    name: str
    adapter: str
    path: Path
    expected_sha256: str | None = None
    expected_records: int | None = None


@dataclass(frozen=True)
class PreparationConfig:
    output_dir: Path
    sources: tuple[SourceConfig, ...]
    ratios: Mapping[str, float]
    seed: int
    minimum_holdout_multiturn: Mapping[str, int]


def load_preparation_config(path: Path) -> PreparationConfig:
    """Load and validate a data-preparation JSON configuration."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be an object")
    source_rows = raw.get("sources")

    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("Configuration requires a non-empty sources list")

    # Sources stay independent so composition and rejection reports remain
    # attributable to ToolACE or xLAM after preparation.
    sources = []
    names = set()

    for row in source_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each source configuration must be an object")

        name = str(row.get("name", "")).strip().lower()
        adapter = str(row.get("adapter", name)).strip().lower()

        if not name or not adapter or name in names:
            raise ValueError(f"Source names and adapters must be non-empty and source names unique: {name!r}")
        names.add(name)

        source_path = _resolve(path.parent, row.get("path"), "source path")
        expected_records = row.get("expected_records")

        if expected_records is not None and (not isinstance(expected_records, int) or expected_records < 1):
            raise ValueError(f"expected_records must be a positive integer for {name}")

        expected_hash = row.get("expected_sha256")

        if expected_hash is not None and (not isinstance(expected_hash, str) or len(expected_hash) != 64):
            raise ValueError(f"expected_sha256 must be a 64-character hash for {name}")

        sources.append(
            SourceConfig(
                name=name,
                adapter=adapter,
                path=source_path,
                expected_sha256=expected_hash,
                expected_records=expected_records,
            )
        )

    output_dir = _resolve(path.parent, raw.get("output_dir"), "output_dir")
    ratios = raw.get("splits", {"train": 0.8, "validation": 0.1, "test": 0.1})
    minimums = raw.get("minimum_holdout_multiturn", {})

    if not isinstance(ratios, Mapping) or not isinstance(minimums, Mapping):
        raise ValueError("splits and minimum_holdout_multiturn must be objects")

    unknown_minimums = set(minimums) - names

    if unknown_minimums:
        raise ValueError(f"Multi-turn minimum configured for unknown sources: {sorted(unknown_minimums)}")

    normalized_minimums = {}

    for name in names:
        value = minimums.get(name, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Multi-turn minimum must be a non-negative integer for {name}")
        normalized_minimums[name] = value

    seed = raw.get("seed", 2026)

    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return PreparationConfig(
        output_dir=output_dir,
        sources=tuple(sources),
        ratios={key: float(value) for key, value in ratios.items()},
        seed=seed,
        minimum_holdout_multiturn=normalized_minimums,
    )


def _convert_source(
    source: SourceConfig, registry: DatasetRegistry
) -> tuple[list[CanonicalRecord], list[dict], int, str, dict]:
    """Convert one source while preserving stable IDs for accepted and rejected rows."""
    adapter = registry.create(source.adapter)
    raw_hash = sha256_file(source.path)
    raw_records = load_raw_records(source.path)

    if source.expected_sha256 is not None and raw_hash != source.expected_sha256:
        raise RuntimeError(f"Source hash mismatch for {source.name}: {raw_hash}")
    if source.expected_records is not None and len(raw_records) != source.expected_records:
        raise RuntimeError(f"Source count mismatch for {source.name}: {len(raw_records)}")

    rows: list[CanonicalRecord] = []
    rejected: list[dict] = []
    source_ids: set[str] = set()

    # Preserve individual rejection reasons for later dataset auditing.
    for index, raw in enumerate(raw_records):
        source_id = adapter.source_id(raw, index)
        if source_id in source_ids:
            raise RuntimeError(f"Duplicate raw source ID for {source.name}: {source_id}")
        source_ids.add(source_id)

        try:
            record = adapter.convert_record(raw, index)
            if record.source != source.name or record.id != source_id:
                raise RuntimeError(f"Adapter identity mismatch for {source_id}")
            rows.append(record)
        except CONVERSION_ERRORS as error:
            # Source data errors are quarantined. Unexpected implementation errors propagate.
            rejected.append({"id": source_id, "source": source.name, "stage": "conversion", "reason": str(error)})

    if len(rows) + len(rejected) != len(raw_records):
        raise RuntimeError(f"Conversion accounting failed for {source.name}")

    reasons = dict(Counter(item["reason"] for item in rejected))
    report = {
        "raw_records": len(raw_records),
        "retained_records": len(rows),
        "rejected_records": len(rejected),
        "rejection_reasons": reasons,
        "raw_source_sha256": raw_hash,
    }
    print(f"CONVERT source={source.name} raw={len(raw_records)} retained={len(rows)} rejected={len(rejected)}")
    if rejected:
        print(f"REJECTIONS source={source.name} reasons={reasons}")

    return rows, rejected, len(raw_records), raw_hash, report


def _write_verified_source(
    source_root: Path,
    source: SourceConfig,
    split_rows: Mapping[str, list[CanonicalRecord]],
    rejected: list[dict],
    raw_count: int,
    raw_hash: str,
) -> None:
    """Write one processed source and verify every serialized record before publication."""
    source_root.mkdir()
    files = {}

    # Reopen and reconstruct every serialized row before publication. This
    # detects lossy serialization as well as malformed canonical records.
    for split in SPLIT_NAMES:
        path = source_root / f"{split}.jsonl"
        expected = [record.to_dict() for record in split_rows[split]]
        files[path.name] = write_jsonl(path, expected)

        reopened = list(read_jsonl(path))
        if reopened != expected:
            raise RuntimeError(f"Round-trip mismatch: {path}")
        for raw in reopened:
            record_from_dict(raw)

    rejected_path = source_root / "rejected.jsonl"
    files[rejected_path.name] = write_jsonl(rejected_path, rejected)
    if list(read_jsonl(rejected_path)) != rejected:
        raise RuntimeError(f"Round-trip mismatch: {rejected_path}")

    written_records = sum(files[f"{split}.jsonl"]["records"] for split in SPLIT_NAMES)
    written_records += files["rejected.jsonl"]["records"]
    if written_records != raw_count:
        raise RuntimeError(f"Final source accounting failed for {source.name}")

    write_json(
        source_root / "manifest.json",
        {
            "schema_version": "1.0",
            "source": source.name,
            "adapter": source.adapter,
            "raw_source": str(source.path),
            "raw_source_sha256": raw_hash,
            "raw_records": raw_count,
            "files": files,
        },
    )


def _publish_prepared_data(
    config: PreparationConfig,
    plan: Any,
    raw_counts: Mapping[str, int],
    raw_hashes: Mapping[str, str],
    report: Mapping[str, Any],
) -> None:
    """Build the complete output in staging and atomically expose it after verification."""
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}-", dir=config.output_dir.parent))

    try:
        for source in config.sources:
            _write_verified_source(
                staging / source.name,
                source,
                plan.rows[source.name],
                plan.rejections[source.name],
                raw_counts[source.name],
                raw_hashes[source.name],
            )

        write_json(staging / "split-report.json", report)
        write_json(
            staging / "preparation.json",
            {
                "schema_version": "1.0",
                "seed": config.seed,
                "splits": dict(config.ratios),
                "minimum_holdout_multiturn": dict(config.minimum_holdout_multiturn),
                "sources": [source.name for source in config.sources],
            },
        )
        # The final directory appears only after every source passes verification.
        staging.replace(config.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare_data(config: PreparationConfig, registry: DatasetRegistry) -> dict:
    """Convert, split, audit, verify, and atomically publish all configured sources."""
    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")

    # Phase 1 converts and fingerprints all sources without writing outputs.
    converted: dict[str, list[CanonicalRecord]] = {}
    rejections: dict[str, list[dict]] = {}
    raw_counts: dict[str, int] = {}
    raw_hashes: dict[str, str] = {}
    conversion_report: dict[str, dict] = {}

    for source in config.sources:
        rows, rejected, raw_count, raw_hash, source_report = _convert_source(source, registry)
        converted[source.name] = rows
        rejections[source.name] = rejected
        raw_counts[source.name] = raw_count
        raw_hashes[source.name] = raw_hash
        conversion_report[source.name] = source_report

    # Phase 2 assigns complete leakage-linked groups and audits the combined
    # split plan before the atomic publication phase begins.
    plan = plan_splits(
        converted,
        rejections,
        raw_counts,
        ratios=config.ratios,
        seed=config.seed,
        minimum_holdout_multiturn=config.minimum_holdout_multiturn,
    )
    audit = audit_split_leakage(plan.rows)
    report = {
        "conversion": conversion_report,
        **plan.report,
        "splits": audit["counts"],
        "leakage_checks": {key: value for key, value in audit.items() if key.startswith("cross_split")},
    }

    _publish_prepared_data(config, plan, raw_counts, raw_hashes, report)
    print(f"VERIFIED output={config.output_dir} records={audit['unique_records']} zero_cross_split_leakage=True")
    return report


def _resolve(base: Path, value: Any, label: str) -> Path:
    return resolve_path(base, value, label)
