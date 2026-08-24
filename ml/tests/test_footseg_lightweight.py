"""TensorFlow-free contract tests for the segmentation train/eval/export path."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import evaluate  # noqa: E402
import export_tflite  # noqa: E402
from footseg import data as footseg_data  # noqa: E402
from footseg.model import build_foot_segmenter, infer_image_size, validate_image_size  # noqa: E402
from footseg.preprocessing import letterbox_dimensions  # noqa: E402


class ImageSizeContractTests(unittest.TestCase):
    def test_valid_sizes(self) -> None:
        self.assertEqual(validate_image_size(32), 32)
        self.assertEqual(validate_image_size(384), 384)
        self.assertEqual(infer_image_size((None, 384, 384, 3)), 384)

    def test_invalid_sizes_fail_before_tensorflow_is_needed(self) -> None:
        for value in (True, 0, -32, 383):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_foot_segmenter(image_size=value, encoder_weights=None)

    def test_invalid_model_input_shapes(self) -> None:
        for shape in (
            (None, None, 384, 3),
            (None, 384, 320, 3),
            (None, 384, 384, 1),
            [(None, 384, 384, 3), (None, 1)],
        ):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                infer_image_size(shape)


class PreprocessingContractTests(unittest.TestCase):
    def test_letterbox_dimensions_match_tensorflow_flooring(self) -> None:
        self.assertEqual(letterbox_dimensions(427, 641, 384), (255, 384, 64, 0))
        self.assertEqual(letterbox_dimensions(641, 427, 384), (384, 255, 0, 64))


class ManifestPathTests(unittest.TestCase):
    def test_relative_paths_prefer_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_directory = root / "dataset"
            project_directory = root / "project"
            for base in (manifest_directory, project_directory):
                (base / "images").mkdir(parents=True)
                (base / "masks").mkdir(parents=True)
                (base / "images" / "sample.png").touch()
                (base / "masks" / "sample.png").touch()
            manifest = manifest_directory / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "split": "val",
                        "image_path": "images/sample.png",
                        "mask_path": "masks/sample.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(footseg_data, "PROJECT_ROOT", project_directory):
                records = footseg_data.load_manifest_records(manifest, "val")

            self.assertEqual(
                Path(records[0]["image_path"]),
                (manifest_directory / "images" / "sample.png").resolve(),
            )
            self.assertEqual(
                Path(records[0]["mask_path"]),
                (manifest_directory / "masks" / "sample.png").resolve(),
            )


class EvaluationContractTests(unittest.TestCase):
    def test_test_split_requires_one_explicit_threshold(self) -> None:
        with self.assertRaises(ValueError):
            evaluate.parse_thresholds(None, "test")
        with self.assertRaises(ValueError):
            evaluate.parse_thresholds("0.4,0.5", "test")
        self.assertEqual(evaluate.parse_thresholds("0.5", "test"), [0.5])

    def test_threshold_values_are_finite_unique_probabilities(self) -> None:
        for specification in ("nan", "inf", "-0.1", "1.1", "0.5,0.5", ""):
            with self.subTest(specification=specification), self.assertRaises(ValueError):
                evaluate.parse_thresholds(specification, "val")

    def test_failed_geometry_produces_strict_json(self) -> None:
        rows = [
            {
                "dice": 0.8,
                "iou": 0.7,
                "precision": 0.9,
                "recall": 0.75,
                "length_error_percent": None,
                "width_error_percent": None,
            },
            {
                "dice": 0.6,
                "iou": 0.5,
                "precision": 0.7,
                "recall": 0.55,
                "length_error_percent": math.nan,
                "width_error_percent": math.nan,
            },
        ]
        summary = evaluate.summarize_threshold(0.5, rows)
        self.assertIsNone(summary["mean_length_error_percent"])
        self.assertEqual(summary["geometry_failure_samples"], 2)
        self.assertEqual(summary["geometry_failure_rate"], 1.0)
        json.dumps(summary, allow_nan=False)


class ExportContractTests(unittest.TestCase):
    def test_valid_tensor_contract(self) -> None:
        input_shape, output_shape = export_tflite.validate_tflite_tensors(
            [{"shape": [1, 384, 384, 3], "dtype": np.float32}],
            [{"shape": [1, 384, 384, 2], "dtype": np.float32}],
        )
        self.assertEqual(input_shape, (1, 384, 384, 3))
        self.assertEqual(output_shape, (1, 384, 384, 2))

    def test_invalid_tensor_contracts(self) -> None:
        valid_input = [{"shape": [1, 384, 384, 3], "dtype": np.float32}]
        invalid_outputs = (
            [{"shape": [1, 384, 384, 1], "dtype": np.float32}],
            [{"shape": [1, 192, 192, 2], "dtype": np.float32}],
            [{"shape": [1, 384, 384, 2], "dtype": np.uint8}],
        )
        for output_details in invalid_outputs:
            with self.subTest(output=output_details), self.assertRaises(RuntimeError):
                export_tflite.validate_tflite_tensors(valid_input, output_details)


if __name__ == "__main__":
    unittest.main()
