from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from ml.footdata.errors import FootDataError
from ml.footdata.manifest import read_manifest, sha256_file, write_manifest
from ml.footdata.package import export_pretrain_package


def _record(index: int, split: str, group: str) -> dict[str, object]:
    stem = f"{index:06d}"
    return {
        "schema_version": 1,
        "sample_id": f"synfoot-v2:{stem}",
        "task": "binary-foot-segmentation",
        "source_id": "synfoot-v2",
        "source_sample_id": f"rgb/rgb/{stem}.png",
        "image_path": f"rgb/rgb/{stem}.png",
        "mask_path": f"mask/mask/{stem}.png",
        "label_path": f"labels/labels_processed/{stem}.json",
        "width": 480,
        "height": 640,
        "group_id": group,
        "group_resolved": True,
        "synthetic": True,
        "license_id": "LicenseRef-SynFoot-Archive-Terms",
        "split": "train",
        "pretrain_split": split,
        "subsets": ["segmentation-valid", "measurement-top35"],
        "camera": {
            "azimuth_deg": float(index * 47 % 360),
            "tilt_deg": 20.0,
        },
        "qa": {
            "valid": True,
            "geometry_eligible": True,
            "touches_border": False,
            "flags": [],
        },
    }


class PretrainPackageTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        records: list[dict[str, object]] = []
        for index in range(12):
            split = "train" if index < 8 else "val"
            group = f"group-{index % 4}" if split == "train" else f"group-{4 + index % 2}"
            record = _record(index, split, group)
            for key in ("image_path", "mask_path"):
                path = root / str(record[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{key}-{index}".encode("ascii"))
            label = root / str(record["label_path"])
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(json.dumps({"foot-id": group}), encoding="utf-8")
            record["image_sha256"] = sha256_file(root / str(record["image_path"]))
            record["mask_sha256"] = sha256_file(root / str(record["mask_path"]))
            record["label_sha256"] = sha256_file(label)
            records.append(record)
        manifest = root / "samples.jsonl"
        write_manifest(records, manifest)
        return manifest

    def test_package_is_capped_deterministic_and_keeps_relative_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            manifest = self._fixture(root)
            first = export_pretrain_package(
                manifest,
                Path(temporary) / "package-a",
                train_limit=5,
                validation_limit=3,
                shard_size_bytes=120,
            )
            second = export_pretrain_package(
                manifest,
                Path(temporary) / "package-b",
                train_limit=5,
                validation_limit=3,
                shard_size_bytes=120,
            )

            self.assertEqual((first.train_count, first.validation_count), (5, 3))
            self.assertEqual(first.manifest_path.read_bytes(), second.manifest_path.read_bytes())
            selected = read_manifest(first.manifest_path)
            self.assertEqual(len(selected), 8)
            self.assertTrue(all("measurement-top35" in row["subsets"] for row in selected))
            archive_names: set[str] = set()
            for shard_path in first.shard_paths:
                with zipfile.ZipFile(shard_path) as archive:
                    archive_names.update(archive.namelist())
            for record in selected:
                self.assertIn(record["image_path"], archive_names)
                self.assertIn(record["mask_path"], archive_names)
                self.assertIn(record["label_path"], archive_names)

            lock = json.loads(first.lock_path.read_text(encoding="utf-8"))
            self.assertEqual(lock["counts"], {"train": 5, "val": 3})
            self.assertEqual(len(lock["shards"]), len(first.shard_paths))

    def test_package_rejects_nonempty_destination_and_unsafe_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            manifest = self._fixture(root)
            destination = Path(temporary) / "package"
            destination.mkdir()
            (destination / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FootDataError, "must be empty"):
                export_pretrain_package(manifest, destination)

            records = read_manifest(manifest)
            records[0]["image_path"] = "../escape.png"
            unsafe = root / "unsafe.jsonl"
            write_manifest(records, unsafe)
            with self.assertRaisesRegex(FootDataError, "unsafe image_path"):
                export_pretrain_package(
                    unsafe,
                    Path(temporary) / "unsafe-package",
                    train_limit=None,
                    validation_limit=None,
                )

    def test_package_rejects_malformed_or_leaky_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            manifest = self._fixture(root)
            records = read_manifest(manifest)

            malformed_records = [dict(record) for record in records]
            malformed_records[0].pop("schema_version")
            malformed = root / "malformed.jsonl"
            write_manifest(malformed_records, malformed)
            with self.assertRaisesRegex(FootDataError, "missing fields: schema_version"):
                export_pretrain_package(
                    malformed,
                    Path(temporary) / "malformed-package",
                    train_limit=None,
                    validation_limit=None,
                )

            leaky_records = [dict(record) for record in records]
            leaky_records[8]["group_id"] = str(leaky_records[0]["group_id"])
            leaky = root / "leaky.jsonl"
            write_manifest(leaky_records, leaky)
            with self.assertRaisesRegex(FootDataError, "pretrain group leakage"):
                export_pretrain_package(
                    leaky,
                    Path(temporary) / "leaky-package",
                    train_limit=None,
                    validation_limit=None,
                )

    def test_package_rejects_selected_asset_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            manifest = self._fixture(root)
            records = read_manifest(manifest)
            corrupted = root / str(records[0]["image_path"])
            corrupted.write_bytes(b"corrupted-after-manifest")

            with self.assertRaisesRegex(FootDataError, "checksum mismatch"):
                export_pretrain_package(
                    manifest,
                    Path(temporary) / "corrupt-package",
                    train_limit=None,
                    validation_limit=None,
                )


if __name__ == "__main__":
    unittest.main()
