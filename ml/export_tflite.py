"""Export a trained Keras segmenter to built-in-only FP16 TensorFlow Lite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from footseg.losses import custom_objects
from footseg.model import validate_image_size


def _tensor_shape(detail: dict[str, object], label: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in detail["shape"])  # type: ignore[union-attr]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {label} tensor shape: {detail.get('shape')!r}") from error
    return shape


def validate_tflite_tensors(
    input_details: list[dict[str, object]],
    output_details: list[dict[str, object]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate the fixed Image Segmenter tensor contract before writing a model."""
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError("MediaPipe Image Segmenter requires one input and one output")

    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = _tensor_shape(input_detail, "input")
    output_shape = _tensor_shape(output_detail, "output")
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[3] != 3:
        raise RuntimeError(f"Expected a [1, height, width, 3] input, got {input_shape}")
    if input_shape[1] != input_shape[2]:
        raise RuntimeError(f"Expected a square input tensor, got {input_shape}")
    try:
        validate_image_size(input_shape[1])
    except ValueError as error:
        raise RuntimeError(
            f"Invalid input tensor shape {input_shape}: {error}"
        ) from error
    expected_output_shape = (1, input_shape[1], input_shape[2], 2)
    if output_shape != expected_output_shape:
        raise RuntimeError(
            f"Expected output shape {expected_output_shape}, got {output_shape}"
        )

    try:
        input_dtype = np.dtype(input_detail["dtype"])
        output_dtype = np.dtype(output_detail["dtype"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("TFLite input/output tensor dtype metadata is invalid") from error
    if input_dtype != np.dtype(np.float32) or output_dtype != np.dtype(np.float32):
        raise RuntimeError(
            "Expected float32 input and output tensors, got "
            f"{input_dtype.name} and {output_dtype.name}"
        )
    return input_shape, output_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/foot_segmenter_fp16.tflite"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is not installed. Use Python 3.11 and install ml/requirements-train.txt"
        ) from error

    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects(),
        compile=False,
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape, output_shape = validate_tflite_tensors(input_details, output_details)
    interpreter.set_tensor(
        input_details[0]["index"], np.zeros(input_shape, dtype=np.float32)
    )
    interpreter.invoke()
    smoke_output = np.asarray(interpreter.get_tensor(output_details[0]["index"]))
    if smoke_output.shape != output_shape or not np.all(np.isfinite(smoke_output)):
        raise RuntimeError(
            "Converted model failed its zero-input smoke test: "
            f"shape={smoke_output.shape}, finite={bool(np.all(np.isfinite(smoke_output)))}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_model)

    sha256 = hashlib.sha256(tflite_model).hexdigest()
    manifest = {
        "model": str(args.output.resolve()),
        "sha256": sha256,
        "bytes": len(tflite_model),
        "input": {
            "shape": list(input_shape),
            "dtype": np.dtype(input_details[0]["dtype"]).name,
            "normalization": {"mean": [127.5], "std": [127.5]},
        },
        "output": {
            "shape": list(output_shape),
            "dtype": np.dtype(output_details[0]["dtype"]).name,
            "labels": ["background", "foot"],
        },
        "metadata_attached": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {args.output} ({len(tflite_model):,} bytes, sha256={sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
