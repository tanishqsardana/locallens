from __future__ import annotations

import unittest

from videosearch.moment_clip import ClipExportConfig, build_label_episode_ranges


class MomentClipTest(unittest.TestCase):
    def test_build_label_episode_ranges(self) -> None:
        tracks = [
            {"track_id": 1, "class": "truck", "frame_idx": 10, "time_sec": 1.0},
            {"track_id": 1, "class": "truck", "frame_idx": 11, "time_sec": 1.1},
            {"track_id": 2, "class": "truck", "frame_idx": 20, "time_sec": 2.0},
            {"track_id": 2, "class": "truck", "frame_idx": 21, "time_sec": 2.1},
            {"track_id": 5, "class": "car", "frame_idx": 30, "time_sec": 3.0},
        ]
        out = build_label_episode_ranges(
            tracks,
            label="truck",
            max_gap_frames=2,
            min_episode_frames=2,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["start_frame"], 10)
        self.assertEqual(out[0]["track_ids"], [1])
        self.assertEqual(out[1]["start_frame"], 20)
        self.assertEqual(out[1]["track_ids"], [2])

    def test_clip_export_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            ClipExportConfig(padding_sec=-1.0).validate()


if __name__ == "__main__":
    unittest.main()
