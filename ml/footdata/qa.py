"""Mask normalization and inexpensive segmentation quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MaskQA:
    valid: bool
    flags: list[str]
    foreground_pixels: int
    foreground_fraction: float
    component_count: int
    largest_component_fraction: float
    touches_border: bool
    bbox_xywh: list[int] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_mask(grayscale: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return uint8 {0,255} data and whether the input was already binary."""

    if grayscale.ndim != 2:
        raise ValueError("Mask must be a two-dimensional grayscale array")
    unique = np.unique(grayscale)
    already_binary = bool(np.all(np.isin(unique, np.array([0, 1, 255]))))
    # Binary masks sometimes mix 1-encoded and 255-encoded foreground pixels.
    # Once the value set is known to be binary, every positive value is
    # foreground; applying the grayscale threshold would incorrectly erase 1s.
    threshold = 0 if already_binary or int(unique[-1]) <= 1 else 127
    normalized = (grayscale > threshold).astype(np.uint8) * 255
    return normalized, already_binary


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.area: list[int] = []

    def add(self) -> int:
        label = len(self.parent)
        self.parent.append(label)
        self.area.append(0)
        return label

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> int:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        if self.area[first_root] < self.area[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.area[first_root] += self.area[second_root]
        self.area[second_root] = 0
        return first_root


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _component_areas(foreground: np.ndarray) -> list[int]:
    """Eight-connected component areas via row runs (fast for silhouette masks)."""

    sets = _DisjointSet()
    previous: list[tuple[int, int, int]] = []
    for row in foreground:
        current: list[tuple[int, int, int]] = []
        previous_cursor = 0
        for start, end in _row_runs(row):
            while previous_cursor < len(previous) and previous[previous_cursor][1] < start - 1:
                previous_cursor += 1
            overlap_labels: list[int] = []
            cursor = previous_cursor
            while cursor < len(previous) and previous[cursor][0] <= end + 1:
                overlap_labels.append(previous[cursor][2])
                cursor += 1

            if overlap_labels:
                label = sets.find(overlap_labels[0])
                for other in overlap_labels[1:]:
                    label = sets.union(label, other)
            else:
                label = sets.add()
            label = sets.find(label)
            sets.area[label] += end - start + 1
            current.append((start, end, label))
        previous = current

    roots = {sets.find(index) for index in range(len(sets.parent))}
    # A root can change after its last run; consolidate areas one final time.
    totals = {root: 0 for root in roots}
    for index, area in enumerate(sets.area):
        if area:
            totals[sets.find(index)] = totals.get(sets.find(index), 0) + area
    return sorted(totals.values(), reverse=True)


def analyze_mask(mask: np.ndarray, *, already_binary: bool = True) -> MaskQA:
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    foreground = mask > 0
    total_pixels = int(foreground.size)
    foreground_pixels = int(np.count_nonzero(foreground))
    fraction = foreground_pixels / total_pixels if total_pixels else 0.0
    flags: list[str] = []
    valid = True

    if not already_binary:
        flags.append("mask_normalized_to_binary")
    if foreground_pixels == 0:
        flags.append("empty_mask")
        valid = False
    elif foreground_pixels == total_pixels:
        flags.append("full_mask")
        valid = False
    if foreground_pixels and (fraction < 0.005 or fraction > 0.95):
        flags.append("foreground_fraction_outlier")

    component_areas = _component_areas(foreground) if foreground_pixels else []
    component_count = len(component_areas)
    largest_fraction = (
        component_areas[0] / foreground_pixels if foreground_pixels and component_areas else 0.0
    )
    if component_count > 1:
        flags.append("multiple_components")
    if component_count > 1 and largest_fraction < 0.95:
        flags.append("fragmented_mask")

    touches_border = bool(
        foreground_pixels
        and (
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
    )
    if touches_border:
        flags.append("mask_touches_border")

    bbox: list[int] | None = None
    if foreground_pixels:
        ys, xs = np.nonzero(foreground)
        minimum_x = int(xs.min())
        maximum_x = int(xs.max())
        minimum_y = int(ys.min())
        maximum_y = int(ys.max())
        bbox = [
            minimum_x,
            minimum_y,
            maximum_x - minimum_x + 1,
            maximum_y - minimum_y + 1,
        ]

    return MaskQA(
        valid=valid,
        flags=flags,
        foreground_pixels=foreground_pixels,
        foreground_fraction=fraction,
        component_count=component_count,
        largest_component_fraction=largest_fraction,
        touches_border=touches_border,
        bbox_xywh=bbox,
    )
