from __future__ import annotations

import ast
import base64
import io
import json
import re
import unittest
import zipfile
from pathlib import Path

from ml.colab.build_notebook import PROJECT_ROOT, source_files


class ColabNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "ml" / "colab" / "Foot_Segmentation_Training.ipynb"
        self.notebook = json.loads(self.path.read_text(encoding="utf-8"))

    def test_notebook_structure_and_code_syntax(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(len(self.notebook["cells"]), 13)
        ids = {cell["id"] for cell in self.notebook["cells"]}
        self.assertEqual(len(ids), len(self.notebook["cells"]))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell:{cell['id']}")

    def test_embedded_archive_matches_pipeline_sources(self) -> None:
        cell = next(
            cell for cell in self.notebook["cells"] if cell["id"] == "embedded-code"
        )
        match = re.search(
            r'SOURCE_ARCHIVE_B64 = "([A-Za-z0-9+/=]+)"',
            "".join(cell["source"]),
        )
        self.assertIsNotNone(match)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as archive:
            expected = {path.as_posix() for path in source_files()}
            self.assertEqual(set(archive.namelist()), expected)
            for relative_path in source_files():
                self.assertEqual(
                    archive.read(relative_path.as_posix()),
                    (PROJECT_ROOT / relative_path).read_bytes(),
                )

    def test_notebook_pins_dataset_and_preserves_outputs(self) -> None:
        source = "\n".join("".join(cell["source"]) for cell in self.notebook["cells"])
        self.assertIn(
            "3c1bfe4a037f60925325421d467e2310bf65a9c8b2485314a0069b6bbe579b07",
            source,
        )
        self.assertIn('drive.mount("/content/drive")', source)
        self.assertIn('item_links.get("content") or item_links.get("self")', source)
        self.assertIn('"--split", "val"', source)
        self.assertIn('"--split", "test"', source)
        self.assertIn('"validation_summary.json"', source)
        self.assertIn('"test_summary.json"', source)
        self.assertIn("files.download(zip_path)", source)


if __name__ == "__main__":
    unittest.main()
