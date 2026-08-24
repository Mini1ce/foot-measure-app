from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ml.footdata import importers as importer_module
from ml.footdata.errors import FootDataError
from ml.footdata.importers import discover_pairs, import_archive, import_dataset
from ml.footdata.manifest import read_manifest, sha256_file, validate_manifest
from ml.footdata.qa import normalize_mask


def _write_image(path: Path, colour: tuple[int, int, int], size: tuple[int, int] = (48, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:, :] = colour
    pixels[0:3, 0:3] = tuple(255 - channel for channel in colour)
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_mask(
    path: Path,
    *,
    size: tuple[int, int] = (48, 64),
    value: int = 255,
    empty: bool = False,
    antialiased: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    if not empty:
        pixels[12:54, 15:34] = value
        if antialiased:
            pixels[11, 15:34] = 80
    Image.fromarray(pixels, mode="L").save(path)


class ImporterTests(unittest.TestCase):
    def _make_footnet(self, root: Path) -> None:
        samples = (
            ("Dorsal", "subject01_frame01", (120, 80, 60)),
            ("Plantar", "subject01_frame02", (121, 80, 60)),
            ("Dorsal", "subject02_frame01", (90, 130, 70)),
        )
        for view, stem, colour in samples:
            _write_image(root / view / "images" / f"{stem}.jpg", colour)
            _write_mask(root / view / "masks" / f"{stem}.png")

    def test_footnet_import_pairs_assets_and_groups_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "processed"
            self._make_footnet(source)

            result = import_dataset("footnet", source, output, license_accepted=True)
            self.assertEqual(len(result.records), 3)
            self.assertEqual(result.errors, [])
            self.assertTrue(all(record["qa"]["valid"] for record in result.records))
            self.assertTrue(all(record["group_resolved"] for record in result.records))
            subject01 = [
                record for record in result.records if "subject01" in record["source_sample_id"]
            ]
            self.assertEqual(len({record["group_id"] for record in subject01}), 1)
            self.assertEqual({record["view"] for record in subject01}, {"dorsal", "plantar"})
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.lock_path.is_file())

            loaded = read_manifest(result.manifest_path)
            report = validate_manifest(loaded, data_root=output, verify_hashes=True)
            self.assertTrue(report["valid"], report["errors"])
            with Image.open(output / loaded[0]["image_path"]) as normalized:
                self.assertEqual(normalized.format, "PNG")
                self.assertFalse(normalized.getexif())

    def test_import_requires_explicit_license_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self._make_footnet(source)
            with self.assertRaisesRegex(FootDataError, "license acceptance"):
                import_dataset("footnet", source, root / "output")

    def test_import_rejects_manifest_lock_path_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self._make_footnet(source)
            collision = root / "same.json"
            with self.assertRaisesRegex(FootDataError, "different paths"):
                import_dataset(
                    "footnet",
                    source,
                    root / "processed",
                    manifest_path=collision,
                    lock_path=collision,
                    license_accepted=True,
                )
            self.assertFalse(collision.exists())

    def test_import_artifacts_must_be_directly_under_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "processed"
            self._make_footnet(source)
            cases = (
                {"manifest_path": output / "nested" / "samples.jsonl"},
                {"lock_path": root / "locks" / "dataset.lock.json"},
            )
            for options in cases:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(FootDataError, "directly under --output-root"):
                        import_dataset(
                            "footnet",
                            source,
                            output,
                            license_accepted=True,
                            **options,
                        )
            self.assertFalse(output.exists())

    def test_decoded_dimension_and_pixel_limits_reject_before_asset_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            dimension_source = root / "dimension-source"
            _write_image(dimension_source / "images" / "subject01.png", (1, 2, 3))
            _write_mask(dimension_source / "masks" / "subject01.png")
            dimension_output = root / "dimension-output"
            with patch.object(importer_module, "MAX_DECODED_DIMENSION", 32):
                dimension_result = import_dataset(
                    "footnet", dimension_source, dimension_output, license_accepted=True
                )
            self.assertEqual(dimension_result.records, [])
            self.assertTrue(
                any("Image decoded dimension limit exceeded" in error for error in dimension_result.errors)
            )
            self.assertFalse((dimension_output / "images").exists())

            pixel_source = root / "pixel-source"
            _write_image(
                pixel_source / "images" / "subject01.png", (1, 2, 3), size=(10, 10)
            )
            _write_mask(pixel_source / "masks" / "subject01.png")
            pixel_output = root / "pixel-output"
            with patch.object(importer_module, "MAX_DECODED_PIXELS", 1_000):
                pixel_result = import_dataset(
                    "footnet", pixel_source, pixel_output, license_accepted=True
                )
            self.assertEqual(pixel_result.records, [])
            self.assertTrue(
                any("Mask decoded pixel limit exceeded" in error for error in pixel_result.errors)
            )
            self.assertFalse((pixel_output / "images").exists())

    def test_mixed_binary_mask_preserves_one_and_255_foreground(self) -> None:
        source = np.array([[0, 1, 255], [1, 0, 255]], dtype=np.uint8)
        normalized, already_binary = normalize_mask(source)
        self.assertTrue(already_binary)
        np.testing.assert_array_equal(
            normalized,
            np.array([[0, 255, 255], [255, 0, 255]], dtype=np.uint8),
        )

    def test_mask_qa_records_normalization_empty_and_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_image(source / "images" / "subject01_good.png", (100, 100, 100))
            _write_mask(source / "masks" / "subject01_good.png", antialiased=True)
            _write_image(source / "images" / "subject02_empty.png", (110, 100, 100))
            _write_mask(source / "masks" / "subject02_empty.png", empty=True)
            _write_image(source / "images" / "subject03_wrong-size.png", (120, 100, 100))
            _write_mask(source / "masks" / "subject03_wrong-size.png", size=(40, 40))

            result = import_dataset(
                "footnet", source, root / "processed", license_accepted=True
            )
            by_name = {Path(record["source_sample_id"]).stem: record for record in result.records}
            self.assertIn("mask_normalized_to_binary", by_name["subject01_good"]["qa"]["flags"])
            self.assertFalse(by_name["subject02_empty"]["qa"]["valid"])
            self.assertIn("empty_mask", by_name["subject02_empty"]["qa"]["flags"])
            self.assertFalse(by_name["subject03_wrong-size"]["qa"]["valid"])
            self.assertIn(
                "image_mask_dimension_mismatch",
                by_name["subject03_wrong-size"]["qa"]["flags"],
            )

    def test_synfoot_uses_underlying_scan_id_and_is_marked_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for stem, colour in (("render001", (90, 90, 120)), ("render002", (91, 90, 120))):
                _write_image(source / "rgb" / f"{stem}.png", colour)
                _write_mask(source / "mask" / f"{stem}.png")
                label = {"foot": {"name": "0009-A"}, "camera": {"fov": 60}}
                label_path = source / "labels" / f"{stem}.json"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(json.dumps(label), encoding="utf-8")

            result = import_dataset(
                "synfoot-v1", source, root / "processed", license_accepted=True
            )
            self.assertEqual(len(result.records), 2)
            self.assertTrue(all(record["synthetic"] for record in result.records))
            self.assertTrue(all(record["group_resolved"] for record in result.records))
            self.assertEqual(len({record["group_id"] for record in result.records}), 1)
            self.assertTrue(
                all(
                    "verify_archive_license_terms" in record["qa"]["flags"]
                    for record in result.records
                )
            )

    def test_real_footnet_layout_keeps_naming_map_as_unresolved_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "FootNet_v1.0"
            cases = (
                ("dataset_lefttop", "lefttop_001", "left_topview_0003.png"),
                ("dataset_leftsole", "leftsole_001", "left_soleview_0003.png"),
                ("dataset_rightside", "rightside_001", "right_sideview_0003.png"),
            )
            for index, (subset, new_stem, _) in enumerate(cases):
                base = source / "dataset" / subset
                _write_image(base / "images" / f"{new_stem}.png", (90 + index, 70, 60))
                _write_mask(base / "masks" / f"{new_stem}.png")

            mapping = source / "naming_map.csv"
            mapping.write_text(
                "subset,old_image,new_image,old_mask,new_mask\n"
                + "".join(
                    f"{subset},{old_name},{new_stem}.png,{old_name},{new_stem}.png\n"
                    for subset, new_stem, old_name in cases
                ),
                encoding="utf-8",
            )
            result = import_dataset(
                "footnet", source, root / "processed", license_accepted=True
            )
            self.assertEqual(len(result.records), 3)
            self.assertTrue(all(not record["group_resolved"] for record in result.records))
            self.assertTrue(all(record["group_method"] == "unresolved" for record in result.records))
            self.assertEqual(
                {record["view"] for record in result.records}, {"dorsal", "plantar", "medial"}
            )
            self.assertEqual(
                {record["laterality"] for record in result.records}, {"left", "right"}
            )
            self.assertEqual(
                {record["provenance"]["publisher_original_image_name"] for record in result.records},
                {old_name for _, _, old_name in cases},
            )

    def test_footnet_reversed_dimensions_apply_known_clockwise_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "FootNet_v1.0" / "dataset" / "dataset_righttop"
            _write_image(source / "images" / "righttop_004.jpg", (80, 100, 120), size=(48, 32))
            _write_mask(source / "masks" / "righttop_004.png", size=(32, 48))

            fixture_sha256 = sha256_file(source / "images" / "righttop_004.jpg")
            with patch.dict(
                importer_module.FOOTNET_CLOCKWISE_ROTATIONS,
                {fixture_sha256: "righttop_004.jpg"},
            ):
                result = import_dataset(
                    "footnet",
                    root / "FootNet_v1.0",
                    root / "processed",
                    license_accepted=True,
                )
            record = result.records[0]
            self.assertTrue(record["qa"]["valid"])
            self.assertEqual((record["width"], record["height"]), (32, 48))
            self.assertIn("image_orientation_corrected", record["qa"]["flags"])
            self.assertEqual(
                record["provenance"]["normalization_transforms"],
                ["footnet_rotate_90_clockwise_for_mask_alignment"],
            )

    def test_unpinned_reversed_footnet_dimensions_are_not_rotated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "FootNet_v1.0" / "dataset" / "dataset_righttop"
            _write_image(source / "images" / "righttop_999.jpg", (80, 100, 120), size=(48, 32))
            _write_mask(source / "masks" / "righttop_999.png", size=(32, 48))
            result = import_dataset(
                "footnet", root / "FootNet_v1.0", root / "processed", license_accepted=True
            )
            record = result.records[0]
            self.assertFalse(record["qa"]["valid"])
            self.assertIn("image_mask_dimension_mismatch", record["qa"]["flags"])
            self.assertNotIn("image_orientation_corrected", record["qa"]["flags"])
            self.assertEqual(record["provenance"]["normalization_transforms"], [])

    def test_reversed_synfoot_dimensions_are_not_rotated_by_guess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_image(source / "rgb" / "render001.png", (80, 100, 120), size=(48, 32))
            _write_mask(source / "mask" / "render001.png", size=(32, 48))
            label_path = source / "labels" / "render001.json"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(json.dumps({"foot": {"name": "scan-a"}}), encoding="utf-8")

            result = import_dataset(
                "synfoot-v1", source, root / "processed", license_accepted=True
            )
            record = result.records[0]
            self.assertFalse(record["qa"]["valid"])
            self.assertIn("image_mask_dimension_mismatch", record["qa"]["flags"])
            self.assertEqual(record["provenance"]["normalization_transforms"], [])

    def test_unresolved_real_subject_is_explicitly_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_image(source / "images" / "unknown.png", (40, 60, 80))
            _write_mask(source / "masks" / "unknown.png")
            result = import_dataset(
                "footnet", source, root / "processed", license_accepted=True
            )
            self.assertFalse(result.records[0]["group_resolved"])
            self.assertIn("missing_group_id_train_only", result.records[0]["qa"]["flags"])

    def test_archive_checksum_is_verified_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture"
            self._make_footnet(source)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in source.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.relative_to(source).as_posix())
            with self.assertRaisesRegex(FootDataError, "SHA-256 mismatch"):
                import_archive(
                    "footnet",
                    archive,
                    root / "processed",
                    expected_sha256="0" * 64,
                    license_accepted=True,
                )
            self.assertFalse((root / "processed").exists())

    def test_discovery_reports_unmatched_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "images" / "subject01_a.png", (1, 2, 3))
            _write_mask(root / "masks" / "subject02_b.png")
            pairs, errors, warnings = discover_pairs(root)
            self.assertEqual(pairs, [])
            self.assertTrue(errors)
            self.assertEqual(len(warnings), 2)

    def test_ambiguous_key_is_excluded_instead_of_pairing_first_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "images" / "sample.png", (1, 2, 3))
            _write_image(root / "rgb" / "sample.jpg", (4, 5, 6))
            _write_mask(root / "masks" / "sample.png")

            pairs, errors, _ = discover_pairs(root)
            self.assertEqual(pairs, [])
            self.assertTrue(any("Ambiguous image key" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
