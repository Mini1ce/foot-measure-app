"""Evaluate masks and geometry across a threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from footseg.data import binarize_raw_mask, load_manifest_records
from footseg.geometry import (
    GEOMETRY_METHOD,
    axis_angle_error_degrees,
    binary_mask_metrics,
    boundary_f1_score,
    mask_dimensions,
)
from footseg.losses import custom_objects
from footseg.model import infer_image_size, validate_image_size
from footseg.preprocessing import letterbox


DEFAULT_THRESHOLDS = "0.25,0.35,0.45,0.50,0.55,0.65,0.75"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/evaluation"))
    parser.add_argument(
        "--image-size",
        type=int,
        help="Expected model size; defaults to the loaded model's input size",
    )
    parser.add_argument(
        "--thresholds",
        help=f"Comma-separated thresholds (validation default: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument("--views", nargs="*")
    parser.add_argument(
        "--split-field",
        choices=("split", "pretrain_split"),
        default="split",
        help="Manifest field used for split selection",
    )
    parser.add_argument("--subsets", nargs="*", help="Optional manifest subset tags")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Deterministically cap matching evaluation samples",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--boundary-tolerance-px",
        type=int,
        default=2,
        help="Boundary F1 tolerance at model resolution",
    )
    return parser.parse_args()


def parse_thresholds(specification: str | None, split: str) -> list[float]:
    if specification is None:
        if split == "test":
            raise ValueError(
                "--split test requires exactly one explicit --thresholds value"
            )
        specification = DEFAULT_THRESHOLDS

    pieces = [piece.strip() for piece in specification.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--thresholds must contain comma-separated numbers")
    try:
        thresholds = [float(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("--thresholds must contain comma-separated numbers") from error
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be finite values between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates")
    if split == "test" and len(thresholds) != 1:
        raise ValueError("--split test permits exactly one fixed threshold")
    return thresholds


def percent_error(predicted: float, target: float) -> float:
    return abs(predicted - target) / max(target, 1e-7) * 100.0


def _finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _finite_or_default(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _is_finite(value: object) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def selection_score(summary: dict[str, object]) -> float:
    mean_dice = summary.get("mean_dice")
    length_error = summary.get("mean_length_error_percent")
    width_error = summary.get("mean_width_error_percent")
    failure_rate = summary.get("geometry_failure_rate")
    return (
        _finite_or_default(mean_dice, 0.0)
        - 0.0025 * _finite_or_default(length_error, 100.0)
        - 0.0015 * _finite_or_default(width_error, 100.0)
        - _finite_or_default(failure_rate, 1.0)
    )


def summarize_threshold(
    threshold: float, rows: list[dict[str, object]]
) -> dict[str, float | int | None]:
    numeric_keys = (
        "dice",
        "iou",
        "precision",
        "recall",
        "boundary_f1",
        "length_error_percent",
        "width_error_percent",
        "angle_error_degrees",
    )
    summary: dict[str, float | int | None] = {
        "threshold": threshold,
        "samples": len(rows),
    }
    for key in numeric_keys:
        values = _finite_values(rows, key)
        summary[f"mean_{key}"] = float(np.mean(values)) if values.size else None
        summary[f"median_{key}"] = float(np.median(values)) if values.size else None
        summary[f"p05_{key}"] = float(np.percentile(values, 5)) if values.size else None
        summary[f"p90_{key}"] = float(np.percentile(values, 90)) if values.size else None
        summary[f"p95_{key}"] = float(np.percentile(values, 95)) if values.size else None

    eligible_geometry = sum(bool(row.get("target_geometry_eligible", True)) for row in rows)
    excluded_geometry = len(rows) - eligible_geometry
    valid_geometry = sum(
        bool(row.get("target_geometry_eligible", True))
        and bool(
            row.get(
                "prediction_geometry_valid",
                _is_finite(row.get("length_error_percent")),
            )
        )
        and _is_finite(row.get("length_error_percent"))
        and _is_finite(row.get("width_error_percent"))
        for row in rows
    )
    failures = eligible_geometry - valid_geometry
    summary["target_geometry_eligible_samples"] = eligible_geometry
    summary["target_geometry_excluded_samples"] = excluded_geometry
    summary["geometry_valid_samples"] = valid_geometry
    summary["geometry_failure_samples"] = failures
    summary["geometry_failure_rate"] = failures / eligible_geometry if eligible_geometry else 1.0
    summary["selection_score"] = selection_score(summary)
    return summary


def main() -> int:
    args = parse_args()
    try:
        thresholds = parse_thresholds(args.thresholds, args.split)
        requested_image_size = (
            validate_image_size(args.image_size) if args.image_size is not None else None
        )
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")
        if args.boundary_tolerance_px < 0:
            raise ValueError("--boundary-tolerance-px must be non-negative")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is not installed. Use Python 3.11 and install ml/requirements-train.txt"
        ) from error

    records = load_manifest_records(
        args.manifest,
        args.split,
        views=set(args.views) if args.views else None,
        subsets=set(args.subsets) if args.subsets else None,
        max_samples=args.max_samples,
        sample_seed=args.seed,
        split_field=args.split_field,
    )
    if not records:
        raise SystemExit("No records matched the evaluation filters")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects(),
        compile=False,
    )
    try:
        model_image_size = infer_image_size(model.input_shape)
    except ValueError as error:
        raise SystemExit(f"Invalid model input: {error}") from error
    if requested_image_size is not None and requested_image_size != model_image_size:
        raise SystemExit(
            f"--image-size {requested_image_size} does not match model input size "
            f"{model_image_size}"
        )
    image_size = requested_image_size or model_image_size
    per_threshold: dict[float, list[dict[str, object]]] = {
        value: [] for value in thresholds
    }

    for index, record in enumerate(records, start=1):
        image = np.asarray(Image.open(record["image_path"]).convert("RGB"))
        target = binarize_raw_mask(
            np.asarray(Image.open(record["mask_path"]).convert("L"))
        )
        model_image = letterbox(image, image_size, mask=False).astype(np.float32)
        model_target = letterbox(
            target.astype(np.uint8) * 255, image_size, mask=True
        ) > 0
        batch_prediction = np.asarray(
            model.predict_on_batch(((model_image - 127.5) / 127.5)[None, ...])
        )
        expected_shape = (1, image_size, image_size, 2)
        if batch_prediction.shape != expected_shape:
            raise RuntimeError(
                f"Expected model output shape {expected_shape}, got {batch_prediction.shape}"
            )
        probabilities = batch_prediction[0, ..., 1]
        target_dimensions = mask_dimensions(model_target)

        for threshold in thresholds:
            prediction = probabilities >= threshold
            metrics = binary_mask_metrics(prediction, model_target)
            metrics["boundary_f1"] = boundary_f1_score(
                prediction,
                model_target,
                tolerance_px=args.boundary_tolerance_px,
            )
            predicted_dimensions = mask_dimensions(prediction)
            row: dict[str, object] = {
                "sample_id": record.get("sample_id", str(index)),
                "threshold": threshold,
                **metrics,
                "target_geometry_eligible": target_dimensions is not None,
                "prediction_geometry_valid": predicted_dimensions is not None,
                "target_length_px": target_dimensions.length_px if target_dimensions else None,
                "target_width_px": target_dimensions.width_px if target_dimensions else None,
                "target_geometry_confidence": target_dimensions.confidence if target_dimensions else None,
                "predicted_length_px": predicted_dimensions.length_px if predicted_dimensions else None,
                "predicted_width_px": predicted_dimensions.width_px if predicted_dimensions else None,
                "predicted_geometry_confidence": predicted_dimensions.confidence if predicted_dimensions else None,
                "target_width_position": target_dimensions.width_position if target_dimensions else None,
                "predicted_width_position": predicted_dimensions.width_position if predicted_dimensions else None,
                "length_error_percent": None,
                "width_error_percent": None,
                "angle_error_degrees": None,
            }
            if predicted_dimensions and target_dimensions:
                row["length_error_percent"] = percent_error(
                    predicted_dimensions.length_px, target_dimensions.length_px
                )
                row["width_error_percent"] = percent_error(
                    predicted_dimensions.width_px, target_dimensions.width_px
                )
                row["angle_error_degrees"] = axis_angle_error_degrees(
                    predicted_dimensions.angle_rad,
                    target_dimensions.angle_rad,
                )
            per_threshold[threshold].append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = [row for rows in per_threshold.values() for row in rows]
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summaries: list[dict[str, float | int | None]] = []
    for threshold, rows in per_threshold.items():
        summaries.append(summarize_threshold(threshold, rows))

    best = max(summaries, key=selection_score)
    report = {
        "split": args.split,
        "model": str(args.model.resolve()),
        "image_size": image_size,
        "boundary_tolerance_px": args.boundary_tolerance_px,
        "geometry_method": GEOMETRY_METHOD,
        "filters": {
            "views": sorted(args.views) if args.views else None,
            "subsets": sorted(args.subsets) if args.subsets else None,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "split_field": args.split_field,
        },
        "summaries": summaries,
        "selected": best,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(best, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
