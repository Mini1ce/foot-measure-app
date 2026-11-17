from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from ml.colab.build_notebook import PROJECT_ROOT, source_files
from ml.colab.build_synfoot_notebook import (
    OUTPUT,
    build_notebook,
    canonical_zip_member_name,
    extract_verified_zip_member,
    validate_packaged_records,
)


class SynFootColabNotebookTests(unittest.TestCase):
    def test_notebook_is_t4_ready_and_code_cells_parse(self) -> None:
        notebook = build_notebook()
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook["metadata"]["colab"]["gpuType"], "T4")
        self.assertEqual(len(notebook["cells"]), 16)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell:{cell['id']}")

    def test_notebook_embeds_current_pipeline_and_expected_training_stages(self) -> None:
        notebook = build_notebook()
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn('"--split-field", "pretrain_split"', source)
        self.assertIn('"--subsets", "measurement-top35"', source)
        self.assertIn('"--initial-model", str(SYNFOOT_RUN_DIR / "best.keras")', source)
        self.assertIn("package.lock.json", source)
        self.assertIn("sha256_file(shard_path)", source)
        self.assertIn("canonical_zip_member_name(info.filename)", source)
        self.assertIn("extract_verified_zip_member(", source)
        self.assertIn("extracted asset set", source)
        self.assertIn('package_validation["counts"]', source)
        self.assertNotIn("extractall(", source)
        self.assertIn("stage_fingerprint.json", source)
        self.assertIn("canonical_json_sha256(stage_spec)", source)
        self.assertIn("write_stage_fingerprint(SYNFOOT_RUN_DIR", source)
        self.assertIn("write_stage_fingerprint(ALL_VIEWS_DIR", source)
        self.assertIn("write_stage_fingerprint(DORSAL_DIR", source)
        self.assertIn("training_provenance.json", source)
        self.assertIn('"embedded_source_archive_sha256"', source)
        self.assertIn('"stage_fingerprints"', source)
        self.assertIn('(PROVENANCE_PATH, PROVENANCE_PATH.name)', source)

        embedded = next(
            cell for cell in notebook["cells"] if cell["id"] == "embedded-code"
        )
        match = re.search(
            r'SOURCE_ARCHIVE_B64 = "([A-Za-z0-9+/=]+)"',
            "".join(embedded["source"]),
        )
        self.assertIsNotNone(match)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {path.as_posix() for path in source_files()},
            )

    def test_zip_member_names_must_already_be_canonical(self) -> None:
        self.assertEqual(canonical_zip_member_name("rgb/rgb_000001.png"), "rgb/rgb_000001.png")
        unsafe_names = (
            "",
            "/rgb/x.png",
            "rgb//x.png",
            "rgb/./x.png",
            "rgb/../x.png",
            "rgb\\x.png",
            "C:/rgb/x.png",
            "rgb/x.png/",
            "rgb/\x00x.png",
        )
        for name in unsafe_names:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                canonical_zip_member_name(name)

    def test_verified_member_extraction_checks_kind_containment_and_hash(self) -> None:
        payload = b"verified image bytes"
        expected_digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "output"
            root.mkdir()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                regular = zipfile.ZipInfo("rgb/sample.png")
                regular.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(regular, payload)
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                name = extract_verified_zip_member(
                    archive, archive.infolist()[0], root, expected_digest
                )
            self.assertEqual(name, "rgb/sample.png")
            self.assertEqual((root / name).read_bytes(), payload)

            bad_root = Path(temporary_directory) / "bad-output"
            bad_root.mkdir()
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    extract_verified_zip_member(
                        archive, archive.infolist()[0], bad_root, "0" * 64
                    )
            self.assertFalse((bad_root / name).exists())

            symlink_buffer = io.BytesIO()
            with zipfile.ZipFile(symlink_buffer, "w") as archive:
                symlink = zipfile.ZipInfo("rgb/link.png")
                symlink.create_system = 3
                symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(symlink, "../outside")
            symlink_buffer.seek(0)
            with zipfile.ZipFile(symlink_buffer) as archive:
                with self.assertRaisesRegex(RuntimeError, "directories/symlinks"):
                    extract_verified_zip_member(
                        archive, archive.infolist()[0], root, None
                    )

    def test_package_records_require_exact_counts_and_disjoint_groups(self) -> None:
        def record(sample_id: str, split: str, group_id: str) -> dict[str, object]:
            return {
                "sample_id": sample_id,
                "pretrain_split": split,
                "split": "train",
                "group_id": group_id,
                "group_resolved": True,
                "synthetic": True,
                "qa": {"valid": True},
                "subsets": ["segmentation-valid", "measurement-top35"],
                "image_path": f"rgb/{sample_id}.png",
                "image_sha256": "1" * 64,
                "mask_path": f"mask/{sample_id}.png",
                "mask_sha256": "2" * 64,
                "label_path": f"labels/{sample_id}.json",
                "label_sha256": "3" * 64,
            }

        records = [record("train-1", "train", "foot-a"), record("val-1", "val", "foot-b")]
        package_lock = {
            "counts": {"train": 1, "val": 1},
            "groups": {"train": ["foot-a"], "val": ["foot-b"]},
            "subset": "measurement-top35",
        }
        validated = validate_packaged_records(records, package_lock)
        self.assertEqual(validated["counts"], {"train": 1, "val": 1})
        self.assertEqual(len(validated["expected_assets"]), 6)

        leaking_records = [records[0], record("val-2", "val", "foot-a")]
        leaking_lock = {
            **package_lock,
            "groups": {"train": ["foot-a"], "val": ["foot-a"]},
        }
        with self.assertRaisesRegex(RuntimeError, "group leakage"):
            validate_packaged_records(leaking_records, leaking_lock)
        with self.assertRaisesRegex(RuntimeError, "counts"):
            validate_packaged_records(
                records, {**package_lock, "counts": {"train": 2, "val": 1}}
            )
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            validate_packaged_records(
                [{**records[0], "synthetic": False}, records[1]], package_lock
            )

    def test_stage_resume_requires_matching_fingerprint_and_checkpoint(self) -> None:
        notebook = build_notebook()
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "training-config")

        def sha256_file(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            namespace = {
                "ARTIFACT_DIR": Path(temporary_directory) / "artifacts",
                "SOURCE_ARCHIVE_SHA256": "a" * 64,
                "package_identity": "b" * 64,
                "manifest_sha256": "c" * 64,
                "verified_shards": [{"filename": "one.zip", "sha256": "d" * 64}],
                "Path": Path,
                "hashlib": hashlib,
                "json": json,
                "os": __import__("os"),
                "subprocess": subprocess,
                "sys": sys,
                "sha256_file": sha256_file,
            }
            exec("".join(cell["source"]), namespace)
            run_dir = Path(temporary_directory) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            (run_dir / "best.keras").write_bytes(b"checkpoint-one")
            stage_spec = namespace["make_stage_spec"](
                "test-stage", ["python", "ml/train.py", "--epochs", "1"], {}, None
            )
            namespace["write_stage_fingerprint"](run_dir, stage_spec)
            self.assertTrue(namespace["completed"](run_dir, stage_spec))

            (run_dir / "best.keras").write_bytes(b"checkpoint-two")
            self.assertFalse(namespace["completed"](run_dir, stage_spec))
            self.assertFalse(
                namespace["completed"](run_dir, {**stage_spec, "stage": "changed"})
            )

    def test_generated_notebook_file_matches_builder(self) -> None:
        generated = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(generated, build_notebook())


if __name__ == "__main__":
    unittest.main()
