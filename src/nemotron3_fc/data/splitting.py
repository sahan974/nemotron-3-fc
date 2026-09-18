"""Deterministic whole-conversation splitting with leakage quarantine."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from nemotron3_fc.data.leakage import is_multiturn, offered_tool_keys, query_template_keys, record_fingerprint, user_query_keys
from nemotron3_fc.data.schema import CanonicalRecord

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitPlan:
    rows: dict[str, dict[str, list[CanonicalRecord]]]
    rejections: dict[str, list[dict]]
    report: dict


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.sizes = [1] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parent[right] = left
        self.sizes[left] += self.sizes[right]


def plan_splits(
    converted: Mapping[str, Sequence[CanonicalRecord]],
    conversion_rejections: Mapping[str, Sequence[dict]],
    raw_counts: Mapping[str, int],
    *,
    ratios: Mapping[str, float],
    seed: int,
    minimum_holdout_multiturn: Mapping[str, int],
) -> SplitPlan:
    """Deduplicate, group leakage-linked records, assign groups, and quarantine template overlap."""
    _validate_ratios(ratios)
    sources = tuple(converted)
    duplicate_rejections = {source: [] for source in sources}
    retained: list[CanonicalRecord] = []
    first_fingerprint: dict[str, str] = {}
    for source in sources:
        for record in converted[source]:
            fingerprint = record_fingerprint(record)
            if fingerprint in first_fingerprint:
                duplicate_rejections[source].append({"id": record.id, "source": source, "stage": "exact_duplicate", "reason": "same_tools_and_messages", "kept_id": first_fingerprint[fingerprint]})
            else:
                first_fingerprint[fingerprint] = record.id
                retained.append(record)

    groups = _linked_groups(retained)
    group_info = [_group_info(group, retained) for group in groups]
    assignments = _assign_groups(group_info, retained, ratios, seed, minimum_holdout_multiturn)

    template_splits: dict[str, set[str]] = {}
    for record, split in zip(retained, assignments):
        for template in query_template_keys(record):
            template_splits.setdefault(template, set()).add(split)
    preferred = {template: next(split for split in SPLIT_NAMES if split in splits) for template, splits in template_splits.items()}

    rows = {source: {split: [] for split in SPLIT_NAMES} for source in sources}
    template_rejections = {source: [] for source in sources}
    for record, split in zip(retained, assignments):
        conflicts = sum(preferred[template] != split for template in query_template_keys(record))
        if conflicts:
            template_rejections[record.source].append({"id": record.id, "source": record.source, "stage": "query_template_overlap", "reason": "template_would_cross_splits", "original_split": split, "conflicting_template_count": conflicts})
        else:
            rows[record.source][split].append(record)

    rejections = {}
    for source in sources:
        rejections[source] = list(conversion_rejections[source]) + duplicate_rejections[source] + template_rejections[source]
        retained_count = sum(len(rows[source][split]) for split in SPLIT_NAMES)
        if retained_count + len(rejections[source]) != raw_counts[source]:
            raise RuntimeError(f"Raw-source accounting failed for {source}")
        if any(not rows[source][split] for split in SPLIT_NAMES):
            raise RuntimeError(f"A split is empty for {source}")
        minimum = minimum_holdout_multiturn.get(source, 0)
        if minimum:
            for split in ("validation", "test"):
                actual = sum(is_multiturn(record) for record in rows[source][split])
                if actual < minimum:
                    raise RuntimeError(f"{source} {split} has {actual} multi-turn records; required {minimum}")

    report = {
        "input_records": {source: len(converted[source]) for source in sources},
        "exact_duplicates_removed": {source: len(duplicate_rejections[source]) for source in sources},
        "template_overlap_records_quarantined": {source: len(template_rejections[source]) for source in sources},
        "leakage_linked_groups": len(groups),
        "largest_leakage_linked_group": max(map(len, groups), default=0),
    }
    return SplitPlan(rows=rows, rejections=rejections, report=report)


def _linked_groups(records: Sequence[CanonicalRecord]) -> list[list[int]]:
    union = _UnionFind(len(records))
    owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = [("query", key) for key in user_query_keys(record)]
        keys.extend(("tool", key) for key in offered_tool_keys(record))
        for key in keys:
            if key in owner:
                union.join(index, owner[key])
            else:
                owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(union.find(index), []).append(index)
    return list(groups.values())


def _group_info(indices: list[int], records: Sequence[CanonicalRecord]) -> dict:
    source_counts = Counter(records[index].source for index in indices)
    multiturn = Counter(records[index].source for index in indices if is_multiturn(records[index]))
    return {"indices": indices, "sources": source_counts, "multiturn": multiturn, "smallest_id": min(records[index].id for index in indices)}


def _assign_groups(group_info: list[dict], records: Sequence[CanonicalRecord], ratios: Mapping[str, float], seed: int, minimums: Mapping[str, int]) -> list[str]:
    assignments: list[str | None] = [None] * len(records)
    source_totals = Counter(record.source for record in records)
    counts = {source: {split: 0 for split in SPLIT_NAMES} for source in source_totals}

    def place(group: dict, split: str) -> None:
        for index in group["indices"]:
            if assignments[index] is not None:
                raise RuntimeError("Leakage-linked group assigned twice")
            assignments[index] = split
        for source, amount in group["sources"].items():
            counts[source][split] += amount

    for source, minimum in minimums.items():
        if minimum <= 0:
            continue
        reserved = {
            split: sum(
                group["multiturn"][source]
                for group in group_info
                if assignments[group["indices"][0]] == split
            )
            for split in ("validation", "test")
        }
        eligible = sorted(
            (group for group in group_info if assignments[group["indices"][0]] is None and 0 < group["multiturn"][source] <= 25),
            key=lambda group: (-group["multiturn"][source], len(group["indices"]), group["smallest_id"]),
        )
        for group in eligible:
            if min(reserved.values()) >= minimum:
                break
            split = min(("validation", "test"), key=lambda candidate: (reserved[candidate], counts[source][candidate], candidate))
            place(group, split)
            reserved[split] += group["multiturn"][source]
        if min(reserved.values()) < minimum:
            raise RuntimeError(f"Not enough independent {source} multi-turn groups for both holdouts: {reserved}")

    constrained = {source for source, minimum in minimums.items() if minimum > 0}
    for group in group_info:
        if assignments[group["indices"][0]] is None and any(group["multiturn"][source] for source in constrained):
            place(group, "train")

    remaining = sorted(
        (group for group in group_info if assignments[group["indices"][0]] is None),
        key=lambda group: (-len(group["indices"]), hashlib.sha256(f"{seed}:{group['smallest_id']}".encode()).hexdigest()),
    )
    for group in remaining:
        def score(candidate: str) -> float:
            return sum(
                (counts[source][split] + (group["sources"][source] if split == candidate else 0) - source_totals[source] * ratios[split]) ** 2 / source_totals[source]
                for source in source_totals for split in SPLIT_NAMES
            )
        place(group, min(SPLIT_NAMES, key=score))
    if any(split is None for split in assignments):
        raise RuntimeError("A leakage-linked group was not assigned")
    return [split for split in assignments if split is not None]


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must define exactly: {', '.join(SPLIT_NAMES)}")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in ratios.values()):
        raise ValueError("Split ratios must be positive numbers")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
