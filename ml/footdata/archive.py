"""Archive inspection and extraction that never trusts member paths."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import UnsafeArchiveError


WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 250_000
    max_member_bytes: int = 2 * 1024**3
    max_total_bytes: int = 100 * 1024**3
    max_compression_ratio: float = 1_500.0


@dataclass(frozen=True)
class InspectedMember:
    info: zipfile.ZipInfo
    relative_path: PurePosixPath


def _safe_member_path(raw_name: str) -> PurePosixPath:
    if not raw_name or "\x00" in raw_name:
        raise UnsafeArchiveError("Archive contains an empty or NUL-containing path")

    normalized = raw_name.replace("\\", "/")
    if normalized.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchiveError(f"Archive member uses an absolute path: {raw_name!r}")

    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"Archive member has an unsafe path: {raw_name!r}")

    for part in path.parts:
        if ":" in part:
            raise UnsafeArchiveError(f"Archive member has a drive/stream path: {raw_name!r}")
        base_name = part.rstrip(" .").split(".", 1)[0].casefold()
        if base_name in WINDOWS_RESERVED_NAMES:
            raise UnsafeArchiveError(f"Archive member uses a reserved Windows name: {raw_name!r}")
    return path


def inspect_zip(
    archive_path: str | os.PathLike[str],
    limits: ArchiveLimits | None = None,
) -> list[InspectedMember]:
    """Validate all members before any output directory is created."""

    archive = Path(archive_path)
    active_limits = limits or ArchiveLimits()
    if archive.suffix.casefold() != ".zip":
        raise UnsafeArchiveError(f"Only .zip archives are supported: {archive}")
    if not archive.is_file():
        raise UnsafeArchiveError(f"Archive does not exist: {archive}")

    inspected: list[InspectedMember] = []
    seen_paths: set[str] = set()
    total_bytes = 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > active_limits.max_members:
                raise UnsafeArchiveError(
                    f"Archive has {len(infos)} members; limit is {active_limits.max_members}"
                )

            for info in infos:
                relative_path = _safe_member_path(info.filename)
                path_key = relative_path.as_posix().casefold()
                if path_key in seen_paths:
                    raise UnsafeArchiveError(
                        f"Archive has duplicate/case-colliding member {info.filename!r}"
                    )
                seen_paths.add(path_key)

                unix_type = (info.external_attr >> 16) & 0o170000
                if unix_type == stat.S_IFLNK:
                    raise UnsafeArchiveError(f"Archive contains a symlink: {info.filename!r}")
                if info.flag_bits & 0x1:
                    raise UnsafeArchiveError(f"Archive contains an encrypted member: {info.filename!r}")
                if info.file_size > active_limits.max_member_bytes:
                    raise UnsafeArchiveError(
                        f"Archive member {info.filename!r} exceeds the per-file size limit"
                    )
                total_bytes += info.file_size
                if total_bytes > active_limits.max_total_bytes:
                    raise UnsafeArchiveError("Archive exceeds the total uncompressed size limit")

                if not info.is_dir() and info.file_size >= 1024**2:
                    ratio = info.file_size / max(1, info.compress_size)
                    if ratio > active_limits.max_compression_ratio:
                        raise UnsafeArchiveError(
                            f"Archive member {info.filename!r} has suspicious compression ratio "
                            f"{ratio:.1f}:1"
                        )
                inspected.append(InspectedMember(info=info, relative_path=relative_path))
    except zipfile.BadZipFile as error:
        raise UnsafeArchiveError(f"Invalid ZIP archive: {archive}") from error
    return inspected


def safe_extract_zip(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    limits: ArchiveLimits | None = None,
) -> Path:
    """Extract a validated ZIP into a new directory using an atomic final move."""

    archive = Path(archive_path).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise UnsafeArchiveError(f"Extraction destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    inspected = inspect_zip(archive, limits)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    ).resolve()

    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in inspected:
                output = staging.joinpath(*member.relative_path.parts).resolve()
                if not output.is_relative_to(staging):
                    raise UnsafeArchiveError(
                        f"Archive member escaped staging directory: {member.info.filename!r}"
                    )
                if member.info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue

                output.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with bundle.open(member.info, "r") as source, output.open("xb") as sink:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > member.info.file_size:
                            raise UnsafeArchiveError(
                                f"Archive member expanded beyond declared size: {member.info.filename!r}"
                            )
                        sink.write(chunk)
                if written != member.info.file_size:
                    raise UnsafeArchiveError(
                        f"Archive member size mismatch: {member.info.filename!r}"
                    )
        os.replace(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
