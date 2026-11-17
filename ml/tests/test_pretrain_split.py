from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ml.footdata.errors import SplitError
from ml.footdata.pretrain import split_synfoot_pretraining


def _record(index: int, group_id: str, *, valid: bool = True) -> dict[str, object]:
    return {
        "sample_id": f"synfoot-v2:{index:06d}",
        "group_id": group_id,
        "group_resolved": True,
        "synthetic": True,
        "split": None,
        "qa": {"valid": valid, "flags": []},
    }


def _eight_group_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    index = 0
    for group_index in range(8):
        for _ in range(3):
            records.append(_record(index, f"foot-{group_index}"))
            index += 1
    return records


class SynFootPretrainSplitTests(unittest.TestCase):
    def test_eight_groups_are_split_six_two_without_weakening_main_split(self) -> None:
        records, _, summary = split_synfoot_pretraining(_eight_group_records(), seed=17)

        self.assertEqual(summary["group_counts"], {"train": 6, "val": 2})
        self.assertTrue(all(record["split"] == "train" for record in records))
        self.assertNotIn("test", {record["pretrain_split"] for record in records})

        group_splits: dict[str, set[str]] = {}
        for record in records:
            group_splits.setdefault(str(record["group_id"]), set()).add(
                str(record["pretrain_split"])
            )
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_assignment_is_deterministic_and_input_order_independent(self) -> None:
        records = _eight_group_records()
        first, first_lock, _ = split_synfoot_pretraining(deepcopy(records), seed=91)
        second, second_lock, _ = split_synfoot_pretraining(
            list(reversed(deepcopy(records))), seed=91
        )

        self.assertEqual(first_lock, second_lock)
        self.assertEqual(
            {record["sample_id"]: record["pretrain_split"] for record in first},
            {record["sample_id"]: record["pretrain_split"] for record in second},
        )

    def test_invalid_qa_is_excluded_without_splitting_its_group(self) -> None:
        records = _eight_group_records()
        invalid = _record(999, "foot-0", valid=False)
        invalid["group_resolved"] = False
        records.append(invalid)

        split, _, summary = split_synfoot_pretraining(records)
        invalid_result = next(record for record in split if record["sample_id"] == invalid["sample_id"])
        self.assertEqual(invalid_result["split"], "excluded")
        self.assertEqual(invalid_result["pretrain_split"], "excluded")
        self.assertEqual(summary["counts"]["excluded"], 1)

    def test_exact_duplicate_images_are_excluded_only_from_pretraining(self) -> None:
        records = _eight_group_records()
        duplicate_ids = {
            records[0]["sample_id"],
            records[3]["sample_id"],
        }
        records[0]["qa"]["flags"] = ["exact_duplicate_image"]
        records[3]["qa"]["flags"] = ["exact_duplicate_image"]

        split, _, summary = split_synfoot_pretraining(records, seed=17)
        duplicates = [
            record for record in split if record["sample_id"] in duplicate_ids
        ]

        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(record["split"] == "train" for record in duplicates))
        self.assertTrue(
            all(record["pretrain_split"] == "excluded" for record in duplicates)
        )
        self.assertEqual(summary["counts"]["excluded"], 2)

    def test_lock_preserves_groups_and_rejects_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "pretrain.lock.json"
            initial = _eight_group_records()
            _, first_lock, _ = split_synfoot_pretraining(
                initial, seed=123, lock_path=lock_path
            )

            extended = [*deepcopy(initial), _record(999, "foot-8")]
            _, second_lock, _ = split_synfoot_pretraining(
                extended, seed=123, lock_path=lock_path
            )
            for group_id, assignment in first_lock["groups"].items():
                self.assertEqual(second_lock["groups"][group_id], assignment)

            with self.assertRaisesRegex(SplitError, "different seed"):
                split_synfoot_pretraining(extended, seed=124, lock_path=lock_path)
            with self.assertRaisesRegex(SplitError, "different train_ratio"):
                split_synfoot_pretraining(
                    extended, seed=123, train_ratio=0.8, lock_path=lock_path
                )

    def test_lock_cannot_leave_an_active_split_without_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "pretrain.lock.json"
            for locked_split, empty_split in (("val", "train"), ("train", "validation")):
                with self.subTest(empty_split=empty_split):
                    lock_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "algorithm": "sha256-ranked-groups-v1",
                                "seed": 17,
                                "train_ratio": 0.75,
                                "validation_ratio": 0.25,
                                "groups": {
                                    f"foot-{index}": locked_split for index in range(8)
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        SplitError, f"zero active {empty_split} groups"
                    ):
                        split_synfoot_pretraining(
                            _eight_group_records(), seed=17, lock_path=lock_path
                        )

    def test_unresolved_or_non_train_synthetic_records_are_rejected(self) -> None:
        records = _eight_group_records()
        records[0]["group_resolved"] = False
        with self.assertRaisesRegex(SplitError, "unresolved group"):
            split_synfoot_pretraining(records)

        records = _eight_group_records()
        records[0]["split"] = "val"
        with self.assertRaisesRegex(SplitError, "train-only"):
            split_synfoot_pretraining(records)


if __name__ == "__main__":
    unittest.main()
