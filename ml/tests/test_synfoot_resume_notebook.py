from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from ml.colab.build_synfoot_resume_notebook import OUTPUT, build_notebook


class SynFootResumeNotebookTests(unittest.TestCase):
    def test_notebook_resumes_from_verified_dorsal_checkpoint(self) -> None:
        notebook = build_notebook()
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('FINE_TUNE_EPOCHS = 0', source)
        self.assertIn('BASE_DORSAL_DIR / "best.keras"', source)
        self.assertIn('"--initial-model", str(BASE_CHECKPOINT)', source)
        self.assertIn('measurement-audit', source)
        self.assertIn('measurement-filter', source)
        self.assertIn('raise RuntimeError("ห้าม fine-tune', source)
        self.assertNotIn('"synfoot-v2-pretrain",\n    SYNFOOT_COMMAND', source)

    def test_all_code_cells_parse(self) -> None:
        for cell in build_notebook()["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_generated_notebook_matches_builder(self) -> None:
        generated = json.loads(Path(OUTPUT).read_text(encoding="utf-8"))
        self.assertEqual(generated, build_notebook())


if __name__ == "__main__":
    unittest.main()
