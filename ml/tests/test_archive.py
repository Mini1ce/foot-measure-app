from __future__ import annotations

import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from ml.footdata.archive import ArchiveLimits, inspect_zip, safe_extract_zip
from ml.footdata.errors import UnsafeArchiveError


class SafeArchiveTests(unittest.TestCase):
    def test_valid_zip_extracts_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("FootNet/images/example.jpg", b"image")
                bundle.writestr("FootNet/masks/example.png", b"mask")

            destination = root / "extracted"
            result = safe_extract_zip(archive, destination)
            self.assertEqual(result, destination.resolve())
            self.assertEqual(
                (destination / "FootNet" / "images" / "example.jpg").read_bytes(), b"image"
            )
            self.assertFalse(any(root.glob(".extracted.staging-*")))

    def test_path_traversal_and_windows_drive_are_rejected(self) -> None:
        unsafe_names = ("../outside.txt", "folder/../../outside.txt", "C:/outside.txt")
        for index, member_name in enumerate(unsafe_names):
            with self.subTest(member_name=member_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / f"unsafe-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(member_name, b"unsafe")
                with self.assertRaises(UnsafeArchiveError):
                    safe_extract_zip(archive, root / "output")
                self.assertFalse((root / "output").exists())
                self.assertFalse((root.parent / "outside.txt").exists())

    def test_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "symlink.zip"
            link = zipfile.ZipInfo("dataset/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(link, "../../outside")
            with self.assertRaisesRegex(UnsafeArchiveError, "symlink"):
                inspect_zip(archive)

    def test_size_and_compression_limits_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "large.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("large.bin", b"0" * (2 * 1024 * 1024))

            with self.assertRaisesRegex(UnsafeArchiveError, "per-file"):
                inspect_zip(archive, ArchiveLimits(max_member_bytes=1024))
            with self.assertRaisesRegex(UnsafeArchiveError, "compression ratio"):
                inspect_zip(
                    archive,
                    ArchiveLimits(
                        max_member_bytes=4 * 1024 * 1024,
                        max_total_bytes=4 * 1024 * 1024,
                        max_compression_ratio=10,
                    ),
                )

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("file.txt", b"new")
            destination = root / "output"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(UnsafeArchiveError, "already exists"):
                safe_extract_zip(archive, destination)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
