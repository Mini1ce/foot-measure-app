"""Manifest reader and TensorFlow input pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_SPLIT_FIELDS = frozenset({"split", "pretrain_split"})
MASK_FOREGROUND_THRESHOLD = 128


def binarize_raw_mask(mask: np.ndarray) -> np.ndarray:
    """Apply the raw grayscale mask contract without requiring TensorFlow."""

    return np.asarray(mask) >= MASK_FOREGROUND_THRESHOLD


def _record_subsets(record: dict[str, Any]) -> set[str]:
    """Return normalized subset tags from supported manifest field shapes."""

    tags: set[str] = set()
    for field in ("subset", "subsets", "subset_tags"):
        value = record.get(field)
        if isinstance(value, str) and value:
            tags.add(value)
        elif isinstance(value, list):
            tags.update(item for item in value if isinstance(item, str) and item)
    return tags


def deterministic_sample(
    records: Iterable[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a stable sample independent of manifest ordering."""

    rows = list(records)
    if max_samples is None:
        return rows
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    def rank(record: dict[str, Any]) -> tuple[bytes, str]:
        sample_id = str(record.get("sample_id", ""))
        digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
        return digest, sample_id

    return sorted(rows, key=rank)[:max_samples]


def load_manifest_records(
    manifest_path: str | Path,
    split: str,
    *,
    views: set[str] | None = None,
    sources: set[str] | None = None,
    subsets: set[str] | None = None,
    synthetic: bool | None = None,
    max_samples: int | None = None,
    sample_seed: int = 20260822,
    split_field: str = "split",
) -> list[dict[str, Any]]:
    """Load valid records and resolve their image/mask paths."""

    if split_field not in ALLOWED_SPLIT_FIELDS:
        choices = ", ".join(sorted(ALLOWED_SPLIT_FIELDS))
        raise ValueError(f"split_field must be one of: {choices}")
    manifest = Path(manifest_path).resolve()
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}: {error}") from error
            if record.get(split_field) != split:
                continue
            if record.get("qa", {}).get("valid") is False:
                continue
            if views and record.get("view") not in views:
                continue
            source_id = record.get("source_id", record.get("source"))
            if sources and source_id not in sources:
                continue
            if subsets and not (_record_subsets(record) & subsets):
                continue
            if synthetic is not None and bool(record.get("synthetic")) is not synthetic:
                continue

            resolved = dict(record)
            for key in ("image_path", "mask_path"):
                path = Path(record[key])
                if not path.is_absolute():
                    manifest_candidate = manifest.parent / path
                    project_candidate = PROJECT_ROOT / path
                    path = manifest_candidate if manifest_candidate.exists() else project_candidate
                resolved[key] = str(path.resolve())
            records.append(resolved)
    return deterministic_sample(records, max_samples=max_samples, seed=sample_seed)


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - exercised only without training deps
        raise RuntimeError(
            "TensorFlow is required. Install ml/requirements-train.txt in Python 3.11."
        ) from error
    return tf


def make_dataset(
    records: Iterable[dict[str, Any]],
    *,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
    geometry_augmentation: bool = False,
):
    """Create a deterministic tf.data pipeline with geometry-safe padding."""

    tf = _require_tensorflow()
    rows = list(records)
    if not rows:
        raise ValueError("No manifest records matched the requested split and filters")
    image_paths = [row["image_path"] for row in rows]
    mask_paths = [row["mask_path"] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.enumerate()

    def projective_transform(image, mask, sample_seeds):
        angle = tf.random.stateless_uniform(
            (), sample_seeds[1], minval=-0.20943951, maxval=0.20943951
        )
        scale = tf.random.stateless_uniform(
            (), sample_seeds[2], minval=0.92, maxval=1.08
        )
        translation = tf.random.stateless_uniform(
            (2,), sample_seeds[3], minval=-0.04, maxval=0.04
        ) * tf.cast(image_size, tf.float32)
        cosine = tf.cos(angle)
        sine = tf.sin(angle)
        inverse_scale = tf.math.reciprocal(scale)
        center = tf.cast(image_size - 1, tf.float32) / 2.0
        translated_center_x = center + translation[0]
        translated_center_y = center + translation[1]
        transform = tf.stack(
            [
                inverse_scale * cosine,
                inverse_scale * sine,
                center
                - inverse_scale * cosine * translated_center_x
                - inverse_scale * sine * translated_center_y,
                -inverse_scale * sine,
                inverse_scale * cosine,
                center
                + inverse_scale * sine * translated_center_x
                - inverse_scale * cosine * translated_center_y,
                0.0,
                0.0,
            ]
        )[None, :]
        output_shape = tf.constant([image_size, image_size], dtype=tf.int32)
        transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=image[None, ...],
            transforms=transform,
            output_shape=output_shape,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            fill_value=0.0,
        )[0]
        transformed_mask = tf.raw_ops.ImageProjectiveTransformV3(
            images=mask[None, ...],
            transforms=transform,
            output_shape=output_shape,
            interpolation="NEAREST",
            fill_mode="CONSTANT",
            fill_value=0.0,
        )[0]
        return transformed_image, transformed_mask

    def decode(index, paths):
        image_path, mask_path = paths
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.convert_image_dtype(image, tf.float32) * 255.0

        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        mask.set_shape([None, None, 1])
        mask = tf.cast(mask >= MASK_FOREGROUND_THRESHOLD, tf.float32)

        image = tf.image.resize_with_pad(image, image_size, image_size, method="bilinear")
        mask = tf.image.resize_with_pad(mask, image_size, image_size, method="nearest")

        if training:
            sample_seed = tf.stack(
                [tf.cast(seed, tf.int32), tf.cast(index % (2**31 - 1), tf.int32)]
            )
            sample_seeds = tf.random.experimental.stateless_split(sample_seed, 8)
            flip_left_right = (
                tf.random.stateless_uniform((), sample_seeds[0]) < 0.5
            )
            image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
            mask = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(mask), lambda: mask)
            if geometry_augmentation:
                image, mask = projective_transform(image, mask, sample_seeds)
            image = tf.image.stateless_random_brightness(
                image, max_delta=20.0, seed=sample_seeds[4]
            )
            image = tf.image.stateless_random_contrast(
                image, lower=0.82, upper=1.18, seed=sample_seeds[5]
            )
            image = tf.image.stateless_random_saturation(
                image, lower=0.85, upper=1.15, seed=sample_seeds[6]
            )
            noise = tf.random.stateless_normal(
                tf.shape(image), seed=sample_seeds[7], stddev=3.0
            )
            image = image + noise

        image = tf.clip_by_value(image, 0.0, 255.0)
        image = (image - 127.5) / 127.5
        mask = tf.cast(mask[..., 0] >= 0.5, tf.int32)
        return image, mask

    dataset = dataset.map(
        decode,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)
