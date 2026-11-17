"""Deterministic, group-safe partitions for synthetic pretraining.

The release split deliberately keeps every synthetic sample out of validation
and test.  This module adds a separate ``pretrain_split`` field so a held-out
set of synthetic foot scans can be used for early stopping without weakening
that release policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .errors import SplitError
from .manifest import write_json


PRETRAIN_SPLIT_ALGORITHM = "sha256-ranked-groups-v1"
PRETRAIN_SPLITS = {"train", "val"}


def _group_rank(seed: int, group_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    return digest, group_id


def _validation_group_count(group_count: int, train_ratio: float) -> int:
    if group_count < 2:
        raise SplitError(
            "Synthetic pretraining needs at least two resolved groups so validation "
            "can hold out an entire group"
        )
    # Round half up rather than relying on Python's banker rounding.  The result
    # is the nearest whole-group approximation to the requested ratio, with at
    # least one group on each side.
    count = math.floor(group_count * (1.0 - train_ratio) + 0.5)
    return min(group_count - 1, max(1, count))


def _read_lock(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"Cannot read pretrain split lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise SplitError(f"Pretrain split lock must contain an object: {path}")
    return value


def _locked_groups(existing: dict[str, Any] | None) -> dict[str, str]:
    if existing is None:
        return {}
    value = existing.get("groups", {})
    if not isinstance(value, dict):
        raise SplitError("Existing pretrain split lock groups must be an object")
    result: dict[str, str] = {}
    for group_id, split in value.items():
        if not isinstance(group_id, str) or not group_id:
            raise SplitError("Existing pretrain split lock contains an invalid group_id")
        if split not in PRETRAIN_SPLITS:
            raise SplitError(
                f"Existing pretrain split lock has invalid split {split!r} "
                f"for group {group_id!r}"
            )
        result[group_id] = split
    return result


def _has_exact_duplicate_flag(record: dict[str, Any]) -> bool:
    qa = record.get("qa")
    flags = qa.get("flags") if isinstance(qa, dict) else None
    return isinstance(flags, list) and "exact_duplicate_image" in flags


def split_synfoot_pretraining(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = 20260825,
    train_ratio: float = 0.75,
    lock_path: str | os.PathLike[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Assign group-safe ``pretrain_split`` values without creating a test set.

    Valid, non-excluded synthetic records remain ``split='train'`` in the main
    release policy. Invalid or already-excluded records receive
    ``pretrain_split='excluded'``. Assignments are made per resolved
    ``group_id`` and frozen in an optional lock file.
    """

    if (
        not isinstance(train_ratio, (int, float))
        or isinstance(train_ratio, bool)
        or not math.isfinite(train_ratio)
        or not 0.0 < train_ratio < 1.0
    ):
        raise SplitError("Pretrain train_ratio must be finite and strictly between 0 and 1")

    materialized = [dict(record) for record in records]
    eligible_groups: set[str] = set()
    for record in materialized:
        sample_id = record.get("sample_id")
        if record.get("synthetic") is not True:
            raise SplitError(f"Record {sample_id!r} is not synthetic SynFoot data")
        main_split = record.get("split")
        if main_split not in {None, "train", "excluded"}:
            raise SplitError(
                f"Synthetic record {sample_id!r} is assigned to main split {main_split!r}; "
                "synthetic release data must remain train-only"
            )
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise SplitError(f"Record {sample_id!r} has no group_id")
        qa_valid = record.get("qa", {}).get("valid") is True
        if (
            qa_valid
            and main_split != "excluded"
            and not _has_exact_duplicate_flag(record)
        ):
            if record.get("group_resolved") is not True:
                raise SplitError(
                    f"Record {sample_id!r} has an unresolved group and cannot be used for "
                    "leakage-safe synthetic validation"
                )
            eligible_groups.add(group_id)

    validation_group_count = _validation_group_count(len(eligible_groups), float(train_ratio))
    lock_destination = Path(lock_path).resolve() if lock_path is not None else None
    existing = _read_lock(lock_destination)
    expected_config = {
        "algorithm": PRETRAIN_SPLIT_ALGORITHM,
        "seed": seed,
        "train_ratio": float(train_ratio),
        "validation_ratio": 1.0 - float(train_ratio),
    }
    if existing is not None:
        for key, value in expected_config.items():
            if existing.get(key) != value:
                raise SplitError(
                    f"Existing pretrain split lock uses different {key}; create a new lock "
                    "instead of silently moving held-out groups"
                )
    assignments = _locked_groups(existing)

    active_locked_validation = sum(
        assignments.get(group_id) == "val" for group_id in eligible_groups
    )
    unassigned = sorted(
        eligible_groups - assignments.keys(),
        key=lambda group_id: _group_rank(seed, group_id),
    )
    validation_needed = max(0, validation_group_count - active_locked_validation)
    if validation_needed > len(unassigned):
        if active_locked_validation == 0 and not unassigned:
            raise SplitError(
                "Existing pretrain split lock leaves zero active validation groups"
            )
        raise SplitError(
            "Existing pretrain split lock cannot satisfy the minimum validation-group policy"
        )
    for index, group_id in enumerate(unassigned):
        assignments[group_id] = "val" if index < validation_needed else "train"

    counts = {"train": 0, "val": 0, "excluded": 0}
    for record in materialized:
        qa_valid = record.get("qa", {}).get("valid") is True
        if not qa_valid or record.get("split") == "excluded":
            record["split"] = "excluded"
            record["pretrain_split"] = "excluded"
        elif _has_exact_duplicate_flag(record):
            # Exact copies are unsuitable for a leakage-sensitive pretraining
            # holdout, but a valid synthetic record remains training-only in
            # the main release policy.
            record["split"] = "train"
            record["pretrain_split"] = "excluded"
        else:
            record["split"] = "train"
            record["pretrain_split"] = assignments[record["group_id"]]
        counts[record["pretrain_split"]] += 1

    active_assignments = {
        group_id: assignments[group_id] for group_id in sorted(eligible_groups)
    }
    group_counts = {
        split: sum(value == split for value in active_assignments.values())
        for split in sorted(PRETRAIN_SPLITS)
    }
    empty_active_splits = [
        split for split in sorted(PRETRAIN_SPLITS) if group_counts[split] == 0
    ]
    if empty_active_splits:
        raise SplitError(
            "Pretrain split lock leaves zero active "
            + " or ".join(empty_active_splits)
            + " groups"
        )
    lock = {
        "schema_version": 1,
        **expected_config,
        "groups": dict(sorted(assignments.items())),
    }
    if lock_destination is not None:
        write_json(lock, lock_destination)
    summary = {
        "record_count": len(materialized),
        "eligible_group_count": len(eligible_groups),
        "counts": counts,
        "group_counts": group_counts,
    }
    return materialized, lock, summary
