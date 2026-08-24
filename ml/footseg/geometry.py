"""Pure NumPy segmentation and geometry metrics.

These functions intentionally avoid OpenCV so dataset validation and unit tests
can run with the small bundled Python runtime used by this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskDimensions:
    length_px: float
    width_px: float
    angle_rad: float
    center_x: float
    center_y: float


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


def mask_dimensions(mask: np.ndarray) -> MaskDimensions | None:
    """Measure a binary silhouette along its principal and minor axes."""

    binary = _as_bool(mask)
    rows, columns = np.nonzero(binary)
    if columns.size < 16:
        return None

    points = np.column_stack((columns.astype(np.float64), rows.astype(np.float64)))
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if major_axis[1] < 0:
        major_axis = -major_axis
    minor_axis = np.array([-major_axis[1], major_axis[0]])

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    # Percentile endpoints suppress isolated one-pixel mask spikes.
    length_px = float(np.percentile(major_projection, 99.5) - np.percentile(major_projection, 0.5))

    # Measure width through the central 70% of the foot and use the widest
    # robust slice. This is closer to the app's forefoot-width measurement than
    # a global rotated bounding box.
    lower = np.percentile(major_projection, 15)
    upper = np.percentile(major_projection, 85)
    central_major = major_projection[(major_projection >= lower) & (major_projection <= upper)]
    central_minor = minor_projection[(major_projection >= lower) & (major_projection <= upper)]
    if central_major.size < 16:
        central_minor = minor_projection
    width_px = float(np.percentile(central_minor, 99.5) - np.percentile(central_minor, 0.5))

    return MaskDimensions(
        length_px=max(0.0, length_px),
        width_px=max(0.0, width_px),
        angle_rad=float(np.arctan2(major_axis[1], major_axis[0])),
        center_x=float(center[0]),
        center_y=float(center[1]),
    )
