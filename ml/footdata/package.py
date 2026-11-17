"""Build deterministic, sharded training packages without copying a whole dataset."""

from __future__ import annotations

import hashlib
import math
import os
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .errors import FootDataError
from .manifest import (
    read_manifest,
    require_valid_manifest,
    sha256_file,
    write_json,
    write_manifest,
)


DEFAULT_TRAIN_LIMIT = 12_000
DEFAULT_VALIDATION_LIMIT = 2_000
DEFAULT_SHARD_SIZE_BYTES = 2 * 1024**3
PACKAGE_ALGORITHM = "sha256-stratified-pretrain-package-v1"


@dataclass(frozen=True)
class PackageResult:
    output_directory: Path
    manifest_path: Path
    lock_path: Path
    shard_paths: tuple[Path, ...]
    train_count: int
    validation_count: int


def _hash_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def _positive_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FootDataError(f"{name} must be a positive integer")
    return value


def _azimuth_bin(record: dict[str, Any]) -> str:
    camera = record.get("camera")
    value = camera.get("azimuth_deg") if isinstance(camera, dict) else None
    try:
        azimuth = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(azimuth):
        return "unknown"
    return str(int((azimuth % 360.0) // 45.0))


def _stratified_sample(
    records: Iterable[dict[str, Any]], *, seed: int, limit: int | None
) -> list[dict[str, Any]]:
    materialized = [dict(record) for record in records]
    if limit is None or len(materialized) <= limit:
        return sorted(materialized, key=lambda record: str(record["sample_id"]))

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in materialized:
        key = f"{record.get('group_id')}|azimuth-{_azimuth_bin(record)}"
        strata[key].append(record)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key, rows in strata.items():
        rows.sort(key=lambda record: _hash_key(seed, str(record["sample_id"])))
        queues[key] = deque(rows)

    stratum_order = sorted(queues, key=lambda key: _hash_key(seed, key))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in stratum_order:
            queue = queues[key]
            if not queue:
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return sorted(selected, key=lambda record: str(record["sample_id"]))


def _resolve_asset(root: Path, record: dict[str, Any], field: str) -> tuple[Path, str]:
    sample_id = str(record.get("sample_id", "unknown"))
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise FootDataError(f"{sample_id}: {field} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise FootDataError(f"{sample_id}: unsafe {field} {value!r}")
    source = root.joinpath(*relative.parts).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FootDataError(f"{sample_id}: missing or unsafe {field}: {normalized}")
    return source, relative.as_posix()


def _selected_records(
    records: Iterable[dict[str, Any]],
    *,
    subset: str,
    seed: int,
    train_limit: int | None,
    validation_limit: int | None,
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for source_record in records:
        record = dict(source_record)
        if record.get("qa", {}).get("valid") is False:
            continue
        subsets = record.get("subsets", [])
        if subset and (not isinstance(subsets, list) or subset not in subsets):
            continue
        pretrain_split = record.get("pretrain_split")
        if pretrain_split in by_split:
            by_split[pretrain_split].append(record)

    selected_train = _stratified_sample(
        by_split["train"], seed=seed, limit=train_limit
    )
    selected_validation = _stratified_sample(
        by_split["val"], seed=seed + 1, limit=validation_limit
    )
    if not selected_train:
        raise FootDataError("No eligible pretrain train records matched the package filters")
    if not selected_validation:
        raise FootDataError("No eligible pretrain validation records matched the package filters")
    return [*selected_train, *selected_validation]


def _require_safe_selection(records: Iterable[dict[str, Any]]) -> None:
    groups: dict[str, set[str]] = {"train": set(), "val": set()}
    for record in records:
        sample_id = str(record.get("sample_id", "unknown"))
        split = record.get("pretrain_split")
        if split not in groups:
            raise FootDataError(f"{sample_id}: selected record has no train/val pretrain split")
        if record.get("qa", {}).get("valid") is not True:
            raise FootDataError(f"{sample_id}: selected record must have qa.valid=true")
        if record.get("synthetic") is not True:
            raise FootDataError(f"{sample_id}: selected record must be synthetic")
        if record.get("group_resolved") is not True:
            raise FootDataError(f"{sample_id}: selected record must have a resolved group")
        if record.get("split") != "train":
            raise FootDataError(f"{sample_id}: selected record must keep the main split as train")
        groups[split].add(str(record["group_id"]))

    overlap = groups["train"] & groups["val"]
    if overlap:
        raise FootDataError(
            "Pretrain group leakage between packaged train and val: "
            + ", ".join(sorted(overlap))
        )


def export_pretrain_package(
    manifest_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    subset: str = "measurement-top35",
    seed: int = 20260825,
    train_limit: int | None = DEFAULT_TRAIN_LIMIT,
    validation_limit: int | None = DEFAULT_VALIDATION_LIMIT,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
) -> PackageResult:
    """Package a deterministic top-view subset into upload-friendly ZIP shards.

    Images, masks, and labels keep their manifest-relative paths. Assets are
    streamed directly from the raw dataset into ZIP shards; no normalized copy
    of the full SynFoot release is created.
    """

    train_limit = _positive_limit(train_limit, "train_limit")
    validation_limit = _positive_limit(validation_limit, "validation_limit")
    if isinstance(shard_size_bytes, bool) or not isinstance(shard_size_bytes, int):
        raise FootDataError("shard_size_bytes must be a positive integer")
    if shard_size_bytes <= 0:
        raise FootDataError("shard_size_bytes must be a positive integer")

    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FootDataError(f"Manifest does not exist: {manifest}")
    dataset_root = manifest.parent
    destination = Path(output_directory).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FootDataError(f"Package output directory must be empty: {destination}")

    records = read_manifest(manifest)
    require_valid_manifest(records)
    selected = _selected_records(
        records,
        subset=subset,
        seed=seed,
        train_limit=train_limit,
        validation_limit=validation_limit,
    )
    _require_safe_selection(selected)
    assets: list[tuple[dict[str, Any], list[tuple[Path, str]]]] = []
    archive_names: set[str] = set()
    for record in selected:
        record_assets: list[tuple[Path, str]] = []
        for field in ("image_path", "mask_path", "label_path"):
            if field == "label_path" and not record.get(field):
                continue
            source, archive_name = _resolve_asset(dataset_root, record, field)
            hash_field = f"{field.removesuffix('_path')}_sha256"
            expected_hash = str(record[hash_field])
            if sha256_file(source).casefold() != expected_hash.casefold():
                raise FootDataError(
                    f"{record['sample_id']}: checksum mismatch for {archive_name}"
                )
            if archive_name in archive_names:
                raise FootDataError(f"Duplicate package asset path: {archive_name}")
            archive_names.add(archive_name)
            record_assets.append((source, archive_name))
        assets.append((record, record_assets))

    destination.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    shard: zipfile.ZipFile | None = None
    partial_path: Path | None = None
    final_path: Path | None = None
    current_size = 0

    def close_shard() -> None:
        nonlocal shard, partial_path, final_path, current_size
        if shard is None or partial_path is None or final_path is None:
            return
        shard.close()
        os.replace(partial_path, final_path)
        shard_paths.append(final_path)
        shard = None
        partial_path = None
        final_path = None
        current_size = 0

    try:
        for _, record_assets in assets:
            record_size = sum(source.stat().st_size for source, _ in record_assets)
            if shard is not None and current_size and current_size + record_size > shard_size_bytes:
                close_shard()
            if shard is None:
                shard_number = len(shard_paths) + 1
                final_path = destination / f"synfoot-v2-pretrain-{shard_number:03d}.zip"
                partial_path = final_path.with_suffix(".zip.part")
                shard = zipfile.ZipFile(
                    partial_path,
                    mode="w",
                    compression=zipfile.ZIP_STORED,
                    allowZip64=True,
                )
            for source, archive_name in record_assets:
                shard.write(source, archive_name)
            current_size += record_size
        close_shard()
    except Exception:
        if shard is not None:
            shard.close()
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise

    output_manifest = destination / "samples.pretrain.jsonl"
    write_manifest(selected, output_manifest)
    output_lock = destination / "package.lock.json"
    train_count = sum(record.get("pretrain_split") == "train" for record in selected)
    validation_count = sum(record.get("pretrain_split") == "val" for record in selected)
    groups: dict[str, set[str]] = {"train": set(), "val": set()}
    for record in selected:
        split = str(record["pretrain_split"])
        groups[split].add(str(record["group_id"]))
    write_json(
        {
            "schema_version": 1,
            "algorithm": PACKAGE_ALGORITHM,
            "source_manifest": manifest.name,
            "source_manifest_sha256": sha256_file(manifest),
            "packaged_manifest": output_manifest.name,
            "packaged_manifest_sha256": sha256_file(output_manifest),
            "subset": subset,
            "seed": seed,
            "limits": {"train": train_limit, "val": validation_limit},
            "counts": {"train": train_count, "val": validation_count},
            "groups": {key: sorted(value) for key, value in groups.items()},
            "shards": [
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in shard_paths
            ],
        },
        output_lock,
    )
    return PackageResult(
        output_directory=destination,
        manifest_path=output_manifest,
        lock_path=output_lock,
        shard_paths=tuple(shard_paths),
        train_count=train_count,
        validation_count=validation_count,
    )
