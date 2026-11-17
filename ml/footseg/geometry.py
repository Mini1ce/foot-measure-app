"""Pure NumPy segmentation and geometry metrics.

These functions intentionally avoid OpenCV so dataset validation and unit tests
can run with the small bundled Python runtime used by this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GEOMETRY_METHOD = "heel-to-toe-cross-sections-v1"


@dataclass(frozen=True)
class MaskDimensions:
    length_px: float
    width_px: float
    angle_rad: float
    center_x: float
    center_y: float
    confidence: float = 0.0
    heel_x: float | None = None
    heel_y: float | None = None
    toe_x: float | None = None
    toe_y: float | None = None
    width_start_x: float | None = None
    width_start_y: float | None = None
    width_end_x: float | None = None
    width_end_y: float | None = None
    width_position: float | None = None
    valid_slice_count: int = 0


@dataclass(frozen=True)
class _WidthSlice:
    position: float
    width: float
    minor_low: float
    minor_high: float


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}")
    return array > 0


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return foreground Dice, IoU, precision, and recall."""

    pred = _as_bool(prediction)
    truth = _as_bool(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} != {truth.shape}")

    intersection = int(np.count_nonzero(pred & truth))
    pred_count = int(np.count_nonzero(pred))
    truth_count = int(np.count_nonzero(truth))
    union = pred_count + truth_count - intersection
    epsilon = 1e-7
    return {
        "dice": (2.0 * intersection + epsilon) / (pred_count + truth_count + epsilon),
        "iou": (intersection + epsilon) / (union + epsilon),
        "precision": (intersection + epsilon) / (pred_count + epsilon),
        "recall": (intersection + epsilon) / (truth_count + epsilon),
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel inner boundary of a binary mask."""

    binary = _as_bool(mask)
    if not binary.any():
        return np.zeros_like(binary)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(binary)
    height, width = binary.shape
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
    return binary & ~eroded


def _dilate_with_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x * offset_x + offset_y * offset_y > radius * radius:
                continue
            start_y = radius + offset_y
            start_x = radius + offset_x
            dilated |= padded[
                start_y : start_y + height,
                start_x : start_x + width,
            ]
    return dilated


def boundary_f1_score(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_px: int = 2,
) -> float:
    """Return boundary F1 with a Euclidean pixel tolerance."""

    pred = _as_bool(prediction)
    truth = _as_bool(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} != {truth.shape}")
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, (int, np.integer)):
        raise ValueError("tolerance_px must be a non-negative integer")
    tolerance_px = int(tolerance_px)
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be a non-negative integer")

    predicted_boundary = _mask_boundary(pred)
    target_boundary = _mask_boundary(truth)
    predicted_count = int(np.count_nonzero(predicted_boundary))
    target_count = int(np.count_nonzero(target_boundary))
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0

    matched_prediction = np.count_nonzero(
        predicted_boundary & _dilate_with_disk(target_boundary, tolerance_px)
    )
    matched_target = np.count_nonzero(
        target_boundary & _dilate_with_disk(predicted_boundary, tolerance_px)
    )
    precision = float(matched_prediction) / predicted_count
    recall = float(matched_target) / target_count
    return 2.0 * precision * recall / max(precision + recall, 1e-7)


def axis_angle_error_degrees(predicted_rad: float, target_rad: float) -> float:
    """Return an undirected principal-axis error in the range 0..90 degrees."""

    if not np.isfinite(predicted_rad) or not np.isfinite(target_rad):
        raise ValueError("axis angles must be finite")
    difference = abs((float(predicted_rad) - float(target_rad)) % np.pi)
    difference = min(difference, np.pi - difference)
    return float(np.degrees(difference))


def mask_dimensions(mask: np.ndarray) -> MaskDimensions | None:
    """Measure heel-to-toe length and a robust perpendicular forefoot width.

    Width is selected from cross-sections in the heel-relative 32--82% region.
    This deliberately avoids the heel and toe tips and mirrors the browser
    measurement geometry more closely than a global rotated bounding box.
    """

    binary = _as_bool(mask)
    rows, columns = np.nonzero(binary)
    if columns.size < 16:
        return None

    points = np.column_stack((columns.astype(np.float64), rows.astype(np.float64)))
    boundary = _mask_boundary(binary)
    boundary_rows, boundary_columns = np.nonzero(boundary)
    boundary_points = np.column_stack(
        (boundary_columns.astype(np.float64), boundary_rows.astype(np.float64))
    )
    if boundary_points.shape[0] < 8:
        return None
    center = boundary_points.mean(axis=0)
    centered_boundary = boundary_points - center
    covariance = np.cov(centered_boundary, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if major_axis[1] < 0:
        major_axis = -major_axis
    minor_axis = np.array([-major_axis[1], major_axis[0]])

    centered = points - center
    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    # Percentile endpoints suppress isolated one-pixel mask spikes.
    major_low = float(np.percentile(major_projection, 0.5))
    major_high = float(np.percentile(major_projection, 99.5))
    length_px = major_high - major_low
    if not np.isfinite(length_px) or length_px <= 0:
        return None

    positions = np.linspace(major_low, major_high, 36)
    spacing = length_px / max(len(positions) - 1, 1)
    half_band = max(0.75, spacing * 0.55)
    slices: list[_WidthSlice] = []
    for position in positions:
        band = np.abs(major_projection - position) <= half_band
        values = minor_projection[band]
        if values.size < 4:
            continue
        minor_low = float(np.percentile(values, 2.0))
        minor_high = float(np.percentile(values, 98.0))
        width = minor_high - minor_low
        if np.isfinite(width) and width > 0:
            slices.append(
                _WidthSlice(
                    position=float(position),
                    width=float(width),
                    minor_low=minor_low,
                    minor_high=minor_high,
                )
            )
    if len(slices) < 12:
        return None

    end_count = max(3, len(slices) // 5)
    low_end_width = float(np.median([item.width for item in slices[:end_count]]))
    high_end_width = float(np.median([item.width for item in slices[-end_count:]]))
    heel_at_low = low_end_width <= high_end_width

    candidates: list[_WidthSlice] = []
    for item in slices:
        normalized = (item.position - major_low) / length_px
        heel_relative = normalized if heel_at_low else 1.0 - normalized
        if 0.32 <= heel_relative <= 0.82:
            candidates.append(item)
    if len(candidates) < 6:
        return None

    candidate_widths = np.asarray([item.width for item in candidates], dtype=np.float64)
    representative_width = float(np.percentile(candidate_widths, 75.0))
    selected = min(candidates, key=lambda item: abs(item.width - representative_width))
    width_px = selected.width
    width_ratio = width_px / length_px
    if not np.isfinite(width_ratio) or not 0.16 <= width_ratio <= 0.62:
        return None

    median_width = float(np.median(candidate_widths))
    mad = float(np.median(np.abs(candidate_widths - median_width)))
    stability = 1.0 - min(1.0, mad / max(median_width, 1e-7) * 2.5)
    coverage = min(1.0, len(slices) / len(positions))
    confidence = float(np.clip(0.65 * stability + 0.35 * coverage, 0.0, 1.0))

    heel_projection = major_low if heel_at_low else major_high
    toe_projection = major_high if heel_at_low else major_low
    heel = center + major_axis * heel_projection
    toe = center + major_axis * toe_projection
    width_center = center + major_axis * selected.position
    width_start = width_center + minor_axis * selected.minor_low
    width_end = width_center + minor_axis * selected.minor_high
    normalized_position = (selected.position - major_low) / length_px
    heel_relative_position = normalized_position if heel_at_low else 1.0 - normalized_position

    return MaskDimensions(
        length_px=max(0.0, length_px),
        width_px=max(0.0, width_px),
        angle_rad=float(np.arctan2(major_axis[1], major_axis[0])),
        center_x=float(center[0]),
        center_y=float(center[1]),
        confidence=confidence,
        heel_x=float(heel[0]),
        heel_y=float(heel[1]),
        toe_x=float(toe[0]),
        toe_y=float(toe[1]),
        width_start_x=float(width_start[0]),
        width_start_y=float(width_start[1]),
        width_end_x=float(width_end[0]),
        width_end_y=float(width_end[1]),
        width_position=float(heel_relative_position),
        valid_slice_count=len(slices),
    )
