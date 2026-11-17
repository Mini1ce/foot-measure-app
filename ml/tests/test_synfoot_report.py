from __future__ import annotations

import unittest

from ml.footdata.report import summarize_synfoot


class SynFootReportTests(unittest.TestCase):
    def test_summary_counts_camera_bins_subsets_and_qa(self) -> None:
        records = [
            {
                "group_id": "foot-a",
                "split": "train",
                "pretrain_split": "train",
                "subsets": ["segmentation-valid", "measurement-top25"],
                "camera": {"tilt_deg": 14.999, "azimuth_deg": 359.9},
                "qa": {
                    "valid": True,
                    "geometry_eligible": True,
                    "flags": ["mask_normalized_to_binary"],
                },
            },
            {
                "group_id": "foot-b",
                "split": "train",
                "pretrain_split": "val",
                "subsets": ["segmentation-valid", "measurement-top35"],
                "camera": {"tilt_from_vertical_deg": 25.0, "azimuth_deg": 45.0},
                "qa": {
                    "valid": True,
                    "geometry_eligible": True,
                    "flags": [],
                },
            },
        ]

        summary = summarize_synfoot(records)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["pretrain_splits"], {"train": 1, "val": 1})
        self.assertEqual(summary["camera_tilt_bins_deg"], {"0-15": 1, "25-35": 1})
        self.assertEqual(summary["camera_azimuth_bins_deg"], {"315-360": 1, "45-90": 1})
        self.assertEqual(summary["geometry_eligible_count"], 2)
        self.assertEqual(summary["camera_invalid_count"], 0)


if __name__ == "__main__":
    unittest.main()
