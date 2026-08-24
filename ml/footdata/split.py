"""Deterministic splits that keep subjects, scans, and duplicates together."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SplitError
from .manifest import write_json


SPLIT_ALGORITHM = "sha256-threshold-group-v1"
ASSIGNABLE_SPLITS = {"train", "val", "test"}


class _Groups:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: str, second: str) -> str:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        keep, merge = sorted((first_root, second_root))
        self.parent[merge] = keep
        return keep


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in self.as_tuple()):
            raise SplitError("Split ratios must be finite and non-negative")
        if not math.isclose(sum(self.as_tuple()), 1.0, rel_tol=0, abs_tol=1e-9):
            raise SplitError("Split ratios must sum to 1.0")
        if self.train <= 0:
            raise SplitError("Train ratio must be greater than zero")

    def as_tuple(self) -> tuple[float, float, float]:
        return self.train, self.val, self.test

    def to_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _fraction(seed: int, unit_id: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / (1 << (8 * len(digest)))


def _new_assignment(seed: int, unit_id: str, ratios: SplitRatios) -> str:
    amount = _fraction(seed, unit_id)
    if amount < ratios.train:
        return "train"
    if amount < ratios.train + ratios.val:
        return "val"
    return "test"


def _read_lock(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"Cannot read split lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise SplitError(f"Split lock must contain an object: {path}")
    return value


def _validated_locked_groups(existing: dict[str, Any] | None) -> dict[str, str]:
    if existing is None:
        return {}
    value = existing.get("groups", {})
    if not isinstance(value, dict):
        raise SplitError("Existing split lock groups must be an object")
    groups: dict[str, str] = {}
    for group_id, split in value.items():
        if not isinstance(group_id, str) or not group_id:
            raise SplitError("Existing split lock contains an invalid group_id")
        if split not in ASSIGNABLE_SPLITS:
            raise SplitError(
                f"Existing split lock has invalid split {split!r} for group {group_id!r}"
            )
        groups[group_id] = split
    return groups


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = 20260822,
    ratios: SplitRatios | None = None,
    lock_path: str | os.PathLike[str] | None = None,
    allow_unresolved_eval: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    active_ratios = ratios or SplitRatios()
    active_ratios.validate()
    materialized = [dict(record) for record in records]
    groups = _Groups()
    hashes: dict[str, list[str]] = {}

    for record in materialized:
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise SplitError(f"Record {record.get('sample_id')!r} has no group_id")
        groups.add(group_id)
        image_hash = record.get("image_sha256")
        if isinstance(image_hash, str) and image_hash:
            hashes.setdefault(image_hash.casefold(), []).append(group_id)

    # Exact duplicates are a second grouping constraint. This union also covers
    # duplicates whose source metadata incorrectly lists different subjects.
    for duplicate_groups in hashes.values():
        first = duplicate_groups[0]
        for other in duplicate_groups[1:]:
            groups.union(first, other)

    components: dict[str, set[str]] = {}
    for group_id in list(groups.parent):
        components.setdefault(groups.find(group_id), set()).add(group_id)

    lock_destination = Path(lock_path).resolve() if lock_path is not None else None
    existing = _read_lock(lock_destination)
    expected_config = {
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "ratios": active_ratios.to_dict(),
        "allow_unresolved_eval": allow_unresolved_eval,
    }
    if existing is not None:
        for key, value in expected_config.items():
            if existing.get(key) != value:
                raise SplitError(
                    f"Existing split lock uses different {key}; create a new lock file "
                    "instead of silently moving held-out samples"
                )
    locked_groups = _validated_locked_groups(existing)

    record_by_group: dict[str, list[dict[str, Any]]] = {}
    for record in materialized:
        record_by_group.setdefault(record["group_id"], []).append(record)

    assignments: dict[str, str] = {}
    for component_root in sorted(components):
        component_groups = sorted(components[component_root])
        component_records = [
            record for group_id in component_groups for record in record_by_group[group_id]
        ]
        locked_splits = {locked_groups[group] for group in component_groups if group in locked_groups}
        if len(locked_splits) > 1:
            raise SplitError(
                "A duplicate-connected component joins groups already frozen in different splits: "
                + ", ".join(component_groups)
            )

        has_synthetic = any(record.get("synthetic") for record in component_records)
        has_unresolved = any(not record.get("group_resolved") for record in component_records)
        force_train = has_synthetic or (has_unresolved and not allow_unresolved_eval)
        if locked_splits:
            assignment = next(iter(locked_splits))
            if force_train and assignment != "train":
                raise SplitError(
                    "Existing lock assigns synthetic/unresolved data outside train for groups: "
                    + ", ".join(component_groups)
                )
        elif force_train:
            assignment = "train"
        else:
            unit_id = "|".join(component_groups)
            assignment = _new_assignment(seed, unit_id, active_ratios)
        for group_id in component_groups:
            assignments[group_id] = assignment
            locked_groups[group_id] = assignment

    counts = {"train": 0, "val": 0, "test": 0, "excluded": 0}
    source_counts: dict[str, dict[str, int]] = {}
    for record in materialized:
        record.pop("split_reliability", None)
        if allow_unresolved_eval and not record.get("group_resolved"):
            record["split_reliability"] = "sample-only"
        if not record.get("qa", {}).get("valid", False):
            split = "excluded"
        else:
            split = assignments[record["group_id"]]
        record["split"] = split
        counts[split] += 1
        per_source = source_counts.setdefault(
            str(record.get("source_id")), {"train": 0, "val": 0, "test": 0, "excluded": 0}
        )
        per_source[split] += 1

    lock = {
        "schema_version": 1,
        **expected_config,
        "groups": dict(sorted(locked_groups.items())),
    }
    if lock_destination is not None:
        write_json(lock, lock_destination)
    summary = {
        "record_count": len(materialized),
        "group_count": len(components),
        "allow_unresolved_eval": allow_unresolved_eval,
        "counts": counts,
        "sources": source_counts,
    }
    return materialized, lock, summary
