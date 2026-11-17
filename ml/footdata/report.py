"""Dependency-light coverage summaries for SynFoot manifests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


TILT_EDGES = (0.0, 15.0, 25.0, 35.0, 45.0, 60.0, 75.0, 90.0)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tilt_bin(value: float) -> str:
    for lower, upper in zip(TILT_EDGES, TILT_EDGES[1:], strict=True):
        if lower <= value < upper:
            return f"{int(lower)}-{int(upper)}"
    if math.isclose(value, TILT_EDGES[-1], abs_tol=1e-9):
        return "75-90"
    return "outside-0-90"


def summarize_synfoot(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize groups, camera coverage, subsets, and mask QA."""

    materialized = list(records)
    groups: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    pretrain_splits: Counter[str] = Counter()
    subsets: Counter[str] = Counter()
    qa_flags: Counter[str] = Counter()
    tilt_bins: Counter[str] = Counter()
    azimuth_bins: Counter[str] = Counter()
    camera_invalid = 0
    geometry_eligible = 0

    for record in materialized:
        groups[str(record.get("group_id", "missing"))] += 1
        splits[str(record.get("split", "unset"))] += 1
        pretrain_splits[str(record.get("pretrain_split", "unset"))] += 1

        tags = record.get("subsets", [])
        if isinstance(tags, list):
            subsets.update(tag for tag in tags if isinstance(tag, str) and tag)

        qa = record.get("qa")
        if isinstance(qa, dict):
            flags = qa.get("flags", [])
            if isinstance(flags, list):
                qa_flags.update(flag for flag in flags if isinstance(flag, str) and flag)
            geometry_eligible += qa.get("geometry_eligible") is True

        camera = record.get("camera")
        if not isinstance(camera, dict):
            camera_invalid += 1
            continue
        tilt = _finite_number(
            camera.get("tilt_deg", camera.get("tilt_from_vertical_deg"))
        )
        azimuth = _finite_number(camera.get("azimuth_deg"))
        if tilt is None or azimuth is None:
            camera_invalid += 1
            continue
        tilt_bins[_tilt_bin(tilt)] += 1
        normalized_azimuth = azimuth % 360.0
        lower = int(normalized_azimuth // 45.0) * 45
        azimuth_bins[f"{lower}-{lower + 45}"] += 1

    return {
        "record_count": len(materialized),
        "group_count": len(groups),
        "groups": dict(sorted(groups.items())),
        "splits": dict(sorted(splits.items())),
        "pretrain_splits": dict(sorted(pretrain_splits.items())),
        "subsets": dict(sorted(subsets.items())),
        "geometry_eligible_count": geometry_eligible,
        "camera_invalid_count": camera_invalid,
        "camera_tilt_bins_deg": dict(sorted(tilt_bins.items())),
        "camera_azimuth_bins_deg": dict(sorted(azimuth_bins.items())),
        "qa_flags": dict(sorted(qa_flags.items())),
    }
