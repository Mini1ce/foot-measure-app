"""Streaming JSONL manifest I/O and consistency validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .errors import ManifestError


REQUIRED_FIELDS = {
    "schema_version",
    "sample_id",
    "task",
    "source_id",
    "source_sample_id",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
    "width",
    "height",
    "group_id",
    "group_resolved",
    "synthetic",
    "license_id",
    "qa",
}
ALLOWED_SPLITS = {None, "train", "val", "test", "excluded"}
ALLOWED_SPLIT_RELIABILITY = {None, "sample-only"}
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(records: Iterable[dict[str, Any]], path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    ordered = sorted((dict(record) for record in records), key=lambda item: item["sample_id"])
    content = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in ordered
    )
    _atomic_text(destination, content)
    return destination


def write_json(value: object, path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_text(destination, content)
    return destination


def read_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ManifestError(f"{source}:{line_number}: record must be an object")
                records.append(value)
    except json.JSONDecodeError as error:
        raise ManifestError(f"Invalid JSONL in {source}: {error}") from error
    return records


def _validate_relative_path(value: object, field: str, sample_id: str) -> str | None:
    if not isinstance(value, str) or not value:
        return f"{sample_id}: {field} must be a non-empty relative path"
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return f"{sample_id}: unsafe {field} {value!r}"
    if any(":" in part for part in path.parts):
        return f"{sample_id}: unsafe {field} {value!r}"
    return None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def validate_manifest(
    records: Iterable[dict[str, Any]],
    *,
    data_root: str | os.PathLike[str] | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    materialized = list(records)
    root = Path(data_root).resolve() if data_root is not None else None
    errors: list[str] = []
    warnings: list[str] = []
    if verify_hashes and root is None:
        errors.append("verify_hashes requires data_root")
    sample_ids: set[str] = set()
    asset_paths: dict[str, tuple[str, str]] = {}
    group_splits: dict[str, str] = {}
    hash_splits: dict[str, str] = {}
    sample_only_eval_counts = {"val": 0, "test": 0}

    for index, record in enumerate(materialized, start=1):
        missing = sorted(REQUIRED_FIELDS - record.keys())
        provisional_id = str(record.get("sample_id", f"record-{index}"))
        if missing:
            errors.append(f"{provisional_id}: missing fields: {', '.join(missing)}")
            continue

        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"record-{index}: sample_id must be a non-empty string")
            continue
        if sample_id in sample_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)

        schema_version = record.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
            errors.append(f"{sample_id}: unsupported schema_version")
        for text_field in ("task", "source_id", "source_sample_id", "license_id"):
            value = record.get(text_field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{sample_id}: {text_field} must be a non-empty string")
        for dimension in ("width", "height"):
            value = record.get(dimension)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{sample_id}: {dimension} must be a positive integer")
        for flag_field in ("synthetic", "group_resolved"):
            if not isinstance(record.get(flag_field), bool):
                errors.append(f"{sample_id}: {flag_field} must be boolean")
        qa = record.get("qa")
        qa_valid = qa.get("valid") if isinstance(qa, dict) else None
        if not isinstance(qa, dict) or not isinstance(qa_valid, bool):
            errors.append(f"{sample_id}: qa.valid must be boolean")

        split = record.get("split")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{sample_id}: unsupported split {split!r}")
        if qa_valid is False and split not in {None, "excluded"}:
            warnings.append(f"{sample_id}: invalid QA record is assigned to {split}")
        if record.get("synthetic") is True and split in {"val", "test"}:
            errors.append(f"{sample_id}: synthetic samples cannot be in {split}")
        reliability = record.get("split_reliability")
        if reliability not in ALLOWED_SPLIT_RELIABILITY:
            errors.append(f"{sample_id}: unsupported split_reliability {reliability!r}")
        if reliability == "sample-only" and record.get("group_resolved") is not False:
            errors.append(f"{sample_id}: sample-only reliability requires an unresolved group")
        if record.get("group_resolved") is False and split in {"val", "test"}:
            if reliability == "sample-only":
                sample_only_eval_counts[split] += 1
            else:
                errors.append(f"{sample_id}: unresolved group cannot be in {split}")

        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            errors.append(f"{sample_id}: group_id must be a non-empty string")
        elif split in {"train", "val", "test"}:
            previous_split = group_splits.setdefault(group_id, split)
            if previous_split != split:
                errors.append(
                    f"group leakage: {group_id!r} appears in {previous_split} and {split}"
                )

        image_hash = record.get("image_sha256")
        if not _is_sha256(image_hash):
            errors.append(f"{sample_id}: image_sha256 must be a SHA-256 hex digest")
        elif split in {"train", "val", "test"}:
            normalized_image_hash = image_hash.casefold()
            previous_split = hash_splits.setdefault(normalized_image_hash, split)
            if previous_split != split:
                errors.append(
                    f"duplicate-image leakage: {normalized_image_hash[:12]} appears in "
                    f"{previous_split} and {split}"
                )
        mask_hash = record.get("mask_sha256")
        if not _is_sha256(mask_hash):
            errors.append(f"{sample_id}: mask_sha256 must be a SHA-256 hex digest")

        for field in ("image_path", "mask_path"):
            path_error = _validate_relative_path(record.get(field), field, sample_id)
            if path_error:
                errors.append(path_error)
                continue
            relative_path = str(record[field]).replace("\\", "/")
            asset_key = relative_path.casefold()
            previous_asset = asset_paths.get(asset_key)
            if previous_asset is not None:
                previous_field, previous_sample = previous_asset
                if previous_field == field:
                    errors.append(f"duplicate {field}: {relative_path}")
                else:
                    errors.append(
                        f"{sample_id}: {relative_path} is reused as {previous_field} "
                        f"for {previous_sample} and {field}"
                    )
            else:
                asset_paths[asset_key] = (field, sample_id)

            if root is not None:
                resolved = root.joinpath(*PurePosixPath(relative_path).parts).resolve()
                if not resolved.is_relative_to(root):
                    errors.append(f"{sample_id}: {field} escapes data root")
                elif not resolved.is_file():
                    errors.append(f"{sample_id}: missing {field}: {relative_path}")
                elif verify_hashes:
                    expected = record[f"{'image' if field == 'image_path' else 'mask'}_sha256"]
                    if sha256_file(resolved).casefold() != str(expected).casefold():
                        errors.append(f"{sample_id}: checksum mismatch for {relative_path}")

    sample_only_total = sum(sample_only_eval_counts.values())
    if sample_only_total:
        warnings.append(
            f"sample-only evaluation includes {sample_only_total} unresolved records "
            f"(val={sample_only_eval_counts['val']}, test={sample_only_eval_counts['test']}); "
            "subject leakage cannot be ruled out"
        )

    return {
        "valid": not errors,
        "record_count": len(materialized),
        "errors": errors,
        "warnings": warnings,
    }


def require_valid_manifest(
    records: Iterable[dict[str, Any]],
    *,
    data_root: str | os.PathLike[str] | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    report = validate_manifest(records, data_root=data_root, verify_hashes=verify_hashes)
    if not report["valid"]:
        raise ManifestError("Manifest validation failed:\n- " + "\n- ".join(report["errors"]))
    return report
