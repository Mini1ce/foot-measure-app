from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.footdata.__main__ import _parser, _print_error, _print_json, main


class CliEncodingTests(unittest.TestCase):
    def test_json_output_survives_cp1252_with_thai_path(self) -> None:
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with patch.object(sys, "stdout", stream):
            _print_json({"path": "C:/Users/example/เดสก์ท็อป/data"})
            stream.flush()
        output = raw.getvalue().decode("cp1252")
        self.assertIn("\\u0e40", output)

    def test_verify_hashes_requires_data_root(self) -> None:
        errors = io.StringIO()
        with patch.object(sys, "stderr", errors):
            result = main(["validate", "missing.jsonl", "--verify-hashes"])
        self.assertEqual(result, 2)
        self.assertIn("--verify-hashes requires --data-root", errors.getvalue())

    def test_split_rejects_output_lock_path_collision_before_writing(self) -> None:
        errors = io.StringIO()
        with patch.object(sys, "stderr", errors):
            result = main(
                [
                    "split",
                    "missing.jsonl",
                    "--output",
                    "same-path.json",
                    "--lock",
                    "same-path.json",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("different paths", errors.getvalue())

    def test_split_artifacts_must_be_directly_beside_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifests" / "samples.jsonl"
            cases = (
                (
                    root / "derived" / "split.jsonl",
                    manifest.parent / "split.lock.json",
                    "Split output must be directly beside",
                ),
                (
                    manifest.parent / "split.jsonl",
                    root / "derived" / "split.lock.json",
                    "Split lock must be directly beside",
                ),
            )
            for output, lock, expected in cases:
                with self.subTest(expected=expected):
                    errors = io.StringIO()
                    with patch.object(sys, "stderr", errors):
                        result = main(
                            [
                                "split",
                                str(manifest),
                                "--output",
                                str(output),
                                "--lock",
                                str(lock),
                            ]
                        )
                    self.assertEqual(result, 2)
                    self.assertIn(expected, errors.getvalue())

    def test_expected_manifest_read_failures_are_clean_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_encoding = root / "invalid-encoding.jsonl"
            invalid_encoding.write_bytes(b"\xff")
            invalid_json = root / "invalid-json.jsonl"
            invalid_json.write_text("{not json}\n", encoding="utf-8")
            cases = (
                (root / "missing.jsonl", "missing.jsonl"),
                (invalid_encoding, "utf-8"),
                (invalid_json, "Invalid JSONL"),
            )
            for manifest, expected in cases:
                with self.subTest(manifest=manifest.name):
                    errors = io.StringIO()
                    with patch.object(sys, "stderr", errors):
                        result = main(["validate", str(manifest)])
                    self.assertEqual(result, 2)
                    self.assertTrue(errors.getvalue().startswith("error: "))
                    self.assertIn(expected, errors.getvalue())
                    self.assertNotIn("Traceback", errors.getvalue())

    def test_unresolved_eval_flag_is_explicit(self) -> None:
        args = _parser().parse_args(
            [
                "split",
                "samples.jsonl",
                "--output",
                "split.jsonl",
                "--lock",
                "split.lock.json",
                "--allow-unresolved-eval",
            ]
        )
        self.assertTrue(args.allow_unresolved_eval)

    def test_error_output_uses_backslash_escape_on_legacy_console(self) -> None:
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with patch.object(sys, "stderr", stream):
            _print_error("path: เดสก์ท็อป")
            stream.flush()
        output = raw.getvalue().decode("cp1252")
        self.assertIn("\\u0e40", output)


if __name__ == "__main__":
    unittest.main()
