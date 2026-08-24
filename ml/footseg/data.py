"""Manifest reader and TensorFlow input pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_manifest_records(
    manifest_path: str | Path,
    split: str,
    *,
    views: set[str] | None = None,
    sources: set[str] | None = None,
    synthetic: bool | None = None,
) -> list[dict[str, Any]]:
    """Load valid records and resolve their image/mask paths."""

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
            if record.get("split") != split:
                continue
            if record.get("qa", {}).get("valid") is False:
                continue
            if views and record.get("view") not in views:
                continue
            source_id = record.get("source_id", record.get("source"))
            if sources and source_id not in sources:
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
    return records


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

    def decode(image_path, mask_path):
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.convert_image_dtype(image, tf.float32) * 255.0

        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        mask.set_shape([None, None, 1])
        mask = tf.cast(mask > 0, tf.float32)

        image = tf.image.resize_with_pad(image, image_size, image_size, method="bilinear")
        mask = tf.image.resize_with_pad(mask, image_size, image_size, method="nearest")

        if training:
            flip_left_right = tf.random.uniform((), seed=seed) < 0.5
            image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
            mask = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(mask), lambda: mask)
            image = tf.image.random_brightness(image, max_delta=20.0, seed=seed + 1)
            image = tf.image.random_contrast(image, lower=0.82, upper=1.18, seed=seed + 2)
            image = tf.image.random_saturation(image, lower=0.85, upper=1.15, seed=seed + 3)
            noise = tf.random.normal(tf.shape(image), stddev=3.0, seed=seed + 4)
            image = image + noise

        image = tf.clip_by_value(image, 0.0, 255.0)
        image = (image - 127.5) / 127.5
        mask = tf.cast(mask[..., 0], tf.int32)
        return image, mask

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)
