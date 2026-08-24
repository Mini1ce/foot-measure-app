from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ml.footdata.errors import SplitError
from ml.footdata.manifest import validate_manifest
from ml.footdata.split import SplitRatios, split_records


def _record(
    index: int,
    group: str,
    *,
    synthetic: bool = False,
    resolved: bool = True,
    valid: bool = True,
    image_hash: str | None = None,
) -> dict[str, object]:
    digest = image_hash or f"{index:064x}"
    return {
        "schema_version": 1,
        "sample_id": f"sample-{index:03d}",
        "task": "binary-foot-segmentation",
        "source_id": "synfoot-v1" if synthetic else "footnet-zenodo-20457253",
        "source_sample_id": f"images/sample-{index:03d}.png",
        "image_path": f"images/sample-{index:03d}.png",
        "mask_path": f"masks/sample-{index:03d}.png",
        "image_sha256": digest,
        "mask_sha256": f"{index + 10_000:064x}",
        "width": 64,
        "height": 64,
        "group_id": group,
        "group_resolved": resolved,
        "synthetic": synthetic,
        "license_id": "CC-BY-4.0",
        "split": None,
        "qa": {"valid": valid, "flags": []},
    }


class ManifestAndSplitTests(unittest.TestCase):
    def test_split_is_deterministic_group_safe_and_real_only_for_holdout(self) -> None:
        records: list[dict[str, object]] = []
        index = 1
        for group_index in range(30):
            group = f"real-group-{group_index:02d}"
            records.append(_record(index, group))
            index += 1
            records.append(_record(index, group))
            index += 1
        records.append(_record(index, "synthetic-scan", synthetic=True))
        index += 1
        records.append(_record(index, "unresolved", resolved=False))
        index += 1
        records.append(_record(index, "invalid", valid=False))

        first, _, summary = split_records(deepcopy(records), seed=42)
        second, _, _ = split_records(list(reversed(deepcopy(records))), seed=42)
        first_splits = {record["sample_id"]: record["split"] for record in first}
        second_splits = {record["sample_id"]: record["split"] for record in second}
        self.assertEqual(first_splits, second_splits)
        self.assertGreater(summary["counts"]["val"], 0)
        self.assertGreater(summary["counts"]["test"], 0)

        per_group: dict[str, set[str]] = {}
        for record in first:
            per_group.setdefault(str(record["group_id"]), set()).add(str(record["split"]))
            if record["synthetic"]:
                self.assertEqual(record["split"], "train")
            if not record["group_resolved"] and record["qa"]["valid"]:
                self.assertEqual(record["split"], "train")
                self.assertNotIn("split_reliability", record)
            if not record["qa"]["valid"]:
                self.assertEqual(record["split"], "excluded")
        self.assertTrue(all(len(splits) == 1 for splits in per_group.values()))
        self.assertTrue(validate_manifest(first)["valid"])

    def test_exact_duplicates_connect_different_groups(self) -> None:
        duplicate_hash = "a" * 64
        records = [
            _record(1, "group-a", image_hash=duplicate_hash),
            _record(2, "group-b", image_hash=duplicate_hash.upper()),
        ]
        split, _, summary = split_records(records, seed=7)
        self.assertEqual(split[0]["split"], split[1]["split"])
        self.assertEqual(summary["group_count"], 1)

    def test_manifest_duplicate_hash_leakage_is_case_insensitive(self) -> None:
        first = _record(1, "group-a", image_hash="a" * 64)
        second = _record(2, "group-b", image_hash="A" * 64)
        first["split"] = "train"
        second["split"] = "test"

        report = validate_manifest([first, second])
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate-image leakage" in error for error in report["errors"]))

    def test_lock_preserves_old_groups_when_data_is_extended(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "splits.lock.json"
            initial = [_record(index, f"group-{index}") for index in range(1, 12)]
            first, _, _ = split_records(initial, seed=123, lock_path=lock_path)
            frozen = {record["group_id"]: record["split"] for record in first}

            extended = [*deepcopy(initial), _record(99, "new-group")]
            second, _, _ = split_records(extended, seed=123, lock_path=lock_path)
            for record in second:
                if record["group_id"] in frozen:
                    self.assertEqual(record["split"], frozen[record["group_id"]])

            with self.assertRaisesRegex(SplitError, "different seed"):
                split_records(extended, seed=124, lock_path=lock_path)

    def test_duplicate_that_joins_conflicting_frozen_groups_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "splits.lock.json"
            initial = [_record(index, f"group-{index}") for index in range(1, 30)]
            first, _, _ = split_records(
                initial,
                seed=88,
                ratios=SplitRatios(0.34, 0.33, 0.33),
                lock_path=lock_path,
            )
            by_split: dict[str, dict[str, object]] = {}
            for record in first:
                by_split.setdefault(str(record["split"]), record)
            selected = list(by_split.values())[:2]
            self.assertEqual(len(selected), 2)
            selected[0]["image_sha256"] = "f" * 64
            selected[1]["image_sha256"] = "f" * 64

            with self.assertRaisesRegex(SplitError, "different splits"):
                split_records(
                    first,
                    seed=88,
                    ratios=SplitRatios(0.34, 0.33, 0.33),
                    lock_path=lock_path,
                )

    def test_manifest_rejects_path_escape_and_leakage(self) -> None:
        first = _record(1, "same-group")
        second = _record(2, "same-group")
        first["split"] = "train"
        second["split"] = "test"
        second["image_path"] = "../outside.png"
        report = validate_manifest([first, second])
        self.assertFalse(report["valid"])
        self.assertTrue(any("group leakage" in error for error in report["errors"]))
        self.assertTrue(any("unsafe image_path" in error for error in report["errors"]))

    def test_manifest_rejects_invalid_metadata_hashes_and_cross_role_path_reuse(self) -> None:
        record = _record(1, "group")
        record["task"] = ""
        record["source_id"] = None
        record["source_sample_id"] = " "
        record["license_id"] = ""
        record["image_sha256"] = "z" * 64
        record["mask_sha256"] = "y" * 64
        record["mask_path"] = record["image_path"]

        report = validate_manifest([record])
        self.assertFalse(report["valid"])
        self.assertTrue(any("task must be" in error for error in report["errors"]))
        self.assertTrue(any("source_id must be" in error for error in report["errors"]))
        self.assertTrue(any("SHA-256" in error for error in report["errors"]))
        self.assertTrue(any("reused as image_path" in error for error in report["errors"]))

    def test_invalid_locked_split_is_a_clean_domain_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "splits.lock.json"
            split_records([_record(1, "group")], lock_path=lock_path)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["groups"]["group"] = "bogus"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            with self.assertRaisesRegex(SplitError, "invalid split 'bogus'"):
                split_records([_record(1, "group")], lock_path=lock_path)

    def test_unresolved_eval_is_explicit_warned_and_pinned_in_lock(self) -> None:
        records = [
            _record(index, f"unresolved-{index}", resolved=False) for index in range(1, 41)
        ]
        records.append(_record(100, "synthetic", synthetic=True))
        split, lock, summary = split_records(
            records,
            seed=17,
            ratios=SplitRatios(0.2, 0.4, 0.4),
            allow_unresolved_eval=True,
        )

        unresolved = [record for record in split if not record["group_resolved"]]
        held_out = [record for record in unresolved if record["split"] in {"val", "test"}]
        self.assertTrue(held_out)
        self.assertTrue(
            all(record["split_reliability"] == "sample-only" for record in unresolved)
        )
        synthetic = next(record for record in split if record["synthetic"])
        self.assertEqual(synthetic["split"], "train")
        self.assertEqual(lock["allow_unresolved_eval"], True)
        self.assertEqual(summary["allow_unresolved_eval"], True)

        report = validate_manifest(split)
        self.assertTrue(report["valid"], report["errors"])
        sample_only_warnings = [
            warning for warning in report["warnings"] if "sample-only evaluation" in warning
        ]
        self.assertEqual(len(sample_only_warnings), 1)
        self.assertIn(f"includes {len(held_out)} unresolved records", sample_only_warnings[0])

    def test_split_lock_rejects_unresolved_policy_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "splits.lock.json"
            records = [_record(1, "unresolved", resolved=False)]
            split_records(records, lock_path=lock_path)
            with self.assertRaisesRegex(SplitError, "different allow_unresolved_eval"):
                split_records(records, lock_path=lock_path, allow_unresolved_eval=True)

    def test_invalid_ratios_fail(self) -> None:
        with self.assertRaisesRegex(SplitError, "sum"):
            split_records([_record(1, "group")], ratios=SplitRatios(0.8, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
