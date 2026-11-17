from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ml.footdata.errors import FootDataError
from ml.footdata.manifest import write_manifest
from ml.footdata.measurement_audit import (
    MeasurementAuditPolicy,
    build_measurement_audit,
    filter_reviewed_records,
    write_reviewed_manifest,
)


def _record(sample_id: str, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "split": split,
        "view": "dorsal",
        "image_path": f"images/{sample_id}.png",
        "mask_path": f"masks/{sample_id}.png",
    }


def _write_assets(root: Path, sample_id: str, *, touches_border: bool = False) -> None:
    image = np.full((96, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((96, 64), dtype=np.uint8)
    start = 0 if touches_border else 8
    mask[start:88, 20:45] = 255
    (root / "images").mkdir(exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    Image.fromarray(image).save(root / "images" / f"{sample_id}.png")
    Image.fromarray(mask).save(root / "masks" / f"{sample_id}.png")


class MeasurementAuditTests(unittest.TestCase):
    def test_report_writes_visuals_and_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record("train-foot", "train"), _record("val-foot", "val")]
            _write_assets(root, "train-foot")
            _write_assets(root, "val-foot", touches_border=True)
            output = root / "audit"

            summary = build_measurement_audit(
                records,
                data_root=root,
                output_directory=output,
                policy=MeasurementAuditPolicy(thumbnail_size=128),
            )

            self.assertEqual(summary["record_count"], 2)
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "measurement-review-template.csv").is_file())
            items = json.loads((output / "items.json").read_text(encoding="utf-8"))
            self.assertEqual(len(items), 2)
            self.assertTrue(all((output / item["thumbnail"]).is_file() for item in items))
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("row.decision", html)
            self.assertIn("measurement-review.csv", html)
            val_item = next(item for item in items if item["sample_id"] == "val-foot")
            self.assertIn("mask-touches-border", val_item["reasons"])

    def test_filter_requires_explicit_complete_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record("train-foot", "train"), _record("val-foot", "val")]
            review = root / "review.csv"
            with review.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sample_id", "decision"))
                writer.writeheader()
                writer.writerow({"sample_id": "train-foot", "decision": "include"})
                writer.writerow({"sample_id": "val-foot", "decision": ""})
            with self.assertRaisesRegex(FootDataError, "requires decision"):
                filter_reviewed_records(records, review_csv=review)

    def test_reviewed_manifest_pins_review_and_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record("train-foot", "train"), _record("val-foot", "val")]
            manifest = root / "samples.jsonl"
            review = root / "review.csv"
            output = root / "samples.reviewed.jsonl"
            lock = root / "samples.reviewed.lock.json"
            write_manifest(records, manifest)
            with review.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sample_id", "decision"))
                writer.writeheader()
                writer.writerows(
                    (
                        {"sample_id": "train-foot", "decision": "include"},
                        {"sample_id": "val-foot", "decision": "include"},
                    )
                )

            result = write_reviewed_manifest(
                records,
                input_manifest=manifest,
                review_csv=review,
                output_manifest=output,
                lock_path=lock,
            )

            self.assertEqual(result["included_records"], 2)
            self.assertEqual(len(result["input_manifest_sha256"]), 64)
            self.assertEqual(len(result["review_csv_sha256"]), 64)
            self.assertTrue(output.is_file())
            self.assertTrue(lock.is_file())


if __name__ == "__main__":
    unittest.main()
