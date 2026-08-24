"""TensorFlow-free preprocessing shared by evaluation and lightweight tests."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def letterbox_dimensions(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """Return ``(height, width, offset_y, offset_x)`` for square letterboxing.

    The resize dimensions use floor, matching ``tf.image.resize_with_pad`` used by
    the training pipeline, while retaining at least one pixel for extreme inputs.
    """
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError("image dimensions and letterbox size must be positive")
    scale = min(size / width, size / height)
    resized_width = max(1, math.floor(width * scale))
    resized_height = max(1, math.floor(height * scale))
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    return resized_height, resized_width, offset_y, offset_x


def letterbox(array: np.ndarray, size: int, *, mask: bool) -> np.ndarray:
    """Resize an image or mask with centered zero padding, without TensorFlow."""
    if mask and array.ndim != 2:
        raise ValueError(f"mask must be a two-dimensional array, got {array.shape!r}")
    if not mask and (array.ndim != 3 or array.shape[2] != 3):
        raise ValueError(f"image must be an HxWx3 RGB array, got {array.shape!r}")

    height, width = array.shape[:2]
    resized_height, resized_width, offset_y, offset_x = letterbox_dimensions(
        height, width, size
    )
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = np.asarray(
        Image.fromarray(array).resize((resized_width, resized_height), resampling)
    )
    canvas_shape = (size, size) if mask else (size, size, 3)
    canvas = np.zeros(canvas_shape, dtype=resized.dtype)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas
