"""TensorFlow-free contract tests for the segmentation train/eval/export path."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import evaluate  # noqa: E402
import export_tflite  # noqa: E402
import train  # noqa: E402
from footseg import data as footseg_data  # noqa: E402
from footseg.geometry import (  # noqa: E402
    axis_angle_error_degrees,
    binary_mask_metrics,
    boundary_f1_score,
    mask_dimensions,
)
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

    def test_raw_mask_threshold_is_128(self) -> None:
        values = np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8)
        actual = footseg_data.binarize_raw_mask(values).astype(np.uint8)
        np.testing.assert_array_equal(actual, np.array([0, 0, 0, 1, 1, 1]))


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

    def test_subsets_and_pretrain_split_filter_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.jsonl"
            rows = [
                {
                    "sample_id": "top",
                    "split": "train",
                    "pretrain_split": "val",
                    "subsets": ["top-25", "top-35"],
                    "image_path": "images/top.png",
                    "mask_path": "masks/top.png",
                },
                {
                    "sample_id": "side",
                    "split": "train",
                    "pretrain_split": "train",
                    "subset": "side",
                    "image_path": "images/side.png",
                    "mask_path": "masks/side.png",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            records = footseg_data.load_manifest_records(
                manifest,
                "val",
                subsets={"top-25"},
                split_field="pretrain_split",
            )

            self.assertEqual([record["sample_id"] for record in records], ["top"])

    def test_sample_cap_is_deterministic_and_order_independent(self) -> None:
        rows = [
            {
                "sample_id": f"sample-{index}",
                "image_path": f"images/{index}.png",
                "mask_path": f"masks/{index}.png",
            }
            for index in range(12)
        ]
        first = footseg_data.deterministic_sample(rows, max_samples=4, seed=73)
        second = footseg_data.deterministic_sample(
            reversed(rows), max_samples=4, seed=73
        )
        self.assertEqual(
            [record["sample_id"] for record in first],
            [record["sample_id"] for record in second],
        )
        self.assertNotEqual(
            [record["sample_id"] for record in first],
            [record["sample_id"] for record in footseg_data.deterministic_sample(
                rows, max_samples=4, seed=74
            )],
        )

    def test_sample_cap_and_split_field_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            footseg_data.deterministic_sample([], max_samples=0, seed=1)
        with self.assertRaises(ValueError):
            footseg_data.deterministic_sample([], max_samples="2", seed=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            footseg_data.load_manifest_records(
                Path("unused.jsonl"), "train", split_field="untrusted"
            )

    def test_training_cli_exposes_pretrain_selection_controls(self) -> None:
        arguments = [
            "train.py",
            "--manifest",
            "samples.jsonl",
            "--split-field",
            "pretrain_split",
            "--subsets",
            "measurement-top35",
            "--train-max-samples",
            "12000",
            "--validation-max-samples",
            "2000",
            "--geometry-augmentation",
        ]
        with mock.patch.object(sys, "argv", arguments):
            parsed = train.parse_args()
        self.assertEqual(parsed.split_field, "pretrain_split")
        self.assertEqual(parsed.subsets, ["measurement-top35"])
        self.assertEqual(parsed.train_max_samples, 12000)
        self.assertEqual(parsed.validation_max_samples, 2000)
        self.assertTrue(parsed.geometry_augmentation)


class GeometryMetricTests(unittest.TestCase):
    def test_boundary_f1_uses_pixel_tolerance(self) -> None:
        target = np.zeros((16, 16), dtype=np.uint8)
        target[4:12, 4:12] = 1
        shifted = np.zeros_like(target)
        shifted[4:12, 5:13] = 1

        self.assertEqual(boundary_f1_score(target, target, tolerance_px=0), 1.0)
        self.assertEqual(boundary_f1_score(shifted, target, tolerance_px=1), 1.0)
        self.assertLess(boundary_f1_score(shifted, target, tolerance_px=0), 1.0)

    def test_boundary_f1_handles_empty_masks_and_bad_tolerance(self) -> None:
        empty = np.zeros((8, 8), dtype=np.uint8)
        present = empty.copy()
        present[2:6, 2:6] = 1
        self.assertEqual(boundary_f1_score(empty, empty), 1.0)
        self.assertEqual(boundary_f1_score(present, empty), 0.0)
        with self.assertRaises(ValueError):
            boundary_f1_score(present, empty, tolerance_px=-1)

    def test_axis_angle_error_is_undirected(self) -> None:
        self.assertAlmostEqual(
            axis_angle_error_degrees(math.radians(5), math.radians(175)),
            10.0,
        )
        self.assertAlmostEqual(
            axis_angle_error_degrees(math.radians(10), math.radians(100)),
            90.0,
        )


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow") is not None,
    "TensorFlow training dependency is not installed",
)
class TensorFlowAugmentationTests(unittest.TestCase):
    def test_geometric_augmentation_keeps_image_and_mask_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            image[12:52, 20:44] = 255
            mask[12:52, 20:44] = 255
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)

            dataset = footseg_data.make_dataset(
                [{"image_path": str(image_path), "mask_path": str(mask_path)}],
                image_size=64,
                batch_size=1,
                training=True,
                seed=19,
                geometry_augmentation=True,
            )
            image_batch, mask_batch = next(iter(dataset))
            restored_image = (image_batch.numpy()[0] + 1.0) * 127.5
            image_foreground = np.mean(restored_image, axis=-1) >= 100.0
            transformed_mask = mask_batch.numpy()[0] > 0
            metrics = binary_mask_metrics(image_foreground, transformed_mask)
            self.assertGreater(metrics["iou"], 0.92)


class EvaluationContractTests(unittest.TestCase):
    def test_cross_section_width_is_rotation_stable(self) -> None:
        mask = np.zeros((180, 180), dtype=np.uint8)
        for row in range(35, 146):
            progress = (row - 35) / 110
            half_width = int(12 + 20 * min(progress / 0.7, 1.0))
            mask[row, 90 - half_width : 91 + half_width] = 1
        rotated = np.asarray(
            Image.fromarray(mask * 255).rotate(31, resample=Image.Resampling.NEAREST)
        ) > 0
        original_dimensions = mask_dimensions(mask)
        rotated_dimensions = mask_dimensions(rotated)
        self.assertIsNotNone(original_dimensions)
        self.assertIsNotNone(rotated_dimensions)
        assert original_dimensions is not None and rotated_dimensions is not None
        self.assertLess(
            abs(original_dimensions.width_px - rotated_dimensions.width_px)
            / original_dimensions.width_px
            * 100.0,
            5.0,
        )
        self.assertGreaterEqual(original_dimensions.width_position or 0.0, 0.32)
        self.assertLessEqual(original_dimensions.width_position or 1.0, 0.82)

    def test_toe_spike_does_not_become_forefoot_width(self) -> None:
        mask = np.zeros((170, 120), dtype=np.uint8)
        mask[30:141, 43:78] = 1
        mask[30:38, 10:111] = 1
        dimensions = mask_dimensions(mask)
        self.assertIsNotNone(dimensions)
        assert dimensions is not None
        self.assertLess(dimensions.width_px, 45.0)
        self.assertGreater(dimensions.confidence, 0.5)

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

    def test_ineligible_target_is_not_counted_as_prediction_failure(self) -> None:
        rows = [
            {
                "target_geometry_eligible": False,
                "prediction_geometry_valid": False,
                "length_error_percent": None,
                "width_error_percent": None,
            },
            {
                "target_geometry_eligible": True,
                "prediction_geometry_valid": True,
                "length_error_percent": 1.0,
                "width_error_percent": 2.0,
            },
        ]
        summary = evaluate.summarize_threshold(0.5, rows)
        self.assertEqual(summary["target_geometry_excluded_samples"], 1)
        self.assertEqual(summary["geometry_failure_samples"], 0)
        self.assertEqual(summary["geometry_failure_rate"], 0.0)

    def test_boundary_and_angle_percentiles_are_reported(self) -> None:
        rows = [
            {
                "boundary_f1": 0.8,
                "angle_error_degrees": 2.0,
                "length_error_percent": 1.0,
                "width_error_percent": 3.0,
            },
            {
                "boundary_f1": 1.0,
                "angle_error_degrees": 6.0,
                "length_error_percent": 2.0,
                "width_error_percent": 5.0,
            },
        ]
        summary = evaluate.summarize_threshold(0.5, rows)
        self.assertAlmostEqual(float(summary["median_boundary_f1"]), 0.9)
        self.assertAlmostEqual(float(summary["p90_boundary_f1"]), 0.98)
        self.assertAlmostEqual(float(summary["p90_angle_error_degrees"]), 5.6)


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
