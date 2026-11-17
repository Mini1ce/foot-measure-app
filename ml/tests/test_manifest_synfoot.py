from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ml.footdata.manifest import validate_manifest


def _record(index: int, group: str, pretrain_split: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_id": f"synfoot-v2:{index:06d}",
        "task": "binary-foot-segmentation",
        "source_id": "synfoot-v2",
        "source_sample_id": f"rgb/rgb/{index:06d}.png",
        "image_path": f"rgb/rgb/{index:06d}.png",
        "mask_path": f"mask/mask/{index:06d}.png",
        "label_path": f"labels/labels_processed/{index:06d}.json",
        "image_sha256": f"{index + 1:064x}",
        "mask_sha256": f"{index + 10_001:064x}",
        "label_sha256": f"{index + 20_001:064x}",
        "width": 480,
        "height": 640,
        "group_id": group,
        "group_resolved": True,
        "synthetic": True,
        "license_id": "LicenseRef-SynFoot-Archive-Terms",
        "split": "train",
        "pretrain_split": pretrain_split,
        "subsets": [
            "segmentation-valid",
            "measurement-top35",
            "measurement-top25",
        ],
        "camera": {"tilt_deg": 20.0, "azimuth_deg": 90.0},
        "qa": {
            "valid": True,
            "geometry_eligible": True,
            "touches_border": False,
            "flags": [],
        },
    }


class SynFootManifestTests(unittest.TestCase):
    def test_valid_grouped_pretrain_metadata(self) -> None:
        report = validate_manifest(
            [_record(1, "foot-a", "train"), _record(2, "foot-b", "val")]
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_pretrain_group_leakage_and_label_pair_are_rejected(self) -> None:
        first = _record(1, "same-foot", "train")
        second = _record(2, "same-foot", "val")
        second.pop("label_sha256")
        report = validate_manifest([first, second])
        self.assertFalse(report["valid"])
        self.assertTrue(any("pretrain group leakage" in error for error in report["errors"]))
        self.assertTrue(any("provided together" in error for error in report["errors"]))

    def test_pretrain_duplicate_image_leakage_is_rejected(self) -> None:
        first = _record(1, "foot-a", "train")
        second = _record(2, "foot-b", "val")
        second["image_sha256"] = str(first["image_sha256"]).upper()

        report = validate_manifest([first, second])

        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                "pretrain duplicate-image leakage" in error
                for error in report["errors"]
            )
        )

    def test_incomplete_label_pair_does_not_crash_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(1, "foot-a", "train")
            record.pop("label_sha256")
            for field in ("image_path", "mask_path", "label_path"):
                asset = root / str(record[field])
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(b"asset")

            report = validate_manifest(
                [record], data_root=root, verify_hashes=True
            )

        self.assertFalse(report["valid"])
        self.assertTrue(any("provided together" in error for error in report["errors"]))

    def test_geometry_subset_invariants_are_rejected(self) -> None:
        missing_parent = _record(1, "foot-a", "train")
        missing_parent["subsets"] = ["segmentation-valid", "measurement-top25"]
        unsafe_geometry = deepcopy(_record(2, "foot-b", "val"))
        unsafe_geometry["qa"]["touches_border"] = True
        too_oblique = deepcopy(_record(3, "foot-c", "train"))
        too_oblique["camera"]["tilt_deg"] = 35.1

        report = validate_manifest([missing_parent, unsafe_geometry, too_oblique])
        self.assertFalse(report["valid"])
        self.assertTrue(any("top25 must also" in error for error in report["errors"]))
        self.assertTrue(any("cannot be geometry eligible" in error for error in report["errors"]))
        self.assertTrue(any("top35 exceeds" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
