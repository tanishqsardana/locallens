from __future__ import annotations

import unittest

from videosearch.moment_clip import ClipExportConfig, build_label_episode_ranges, resolve_color_track_ids


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

    def test_build_label_episode_ranges_union_mode(self) -> None:
        tracks = [
            {"track_id": 10, "class": "car", "frame_idx": 100, "time_sec": 10.0},
            {"track_id": 10, "class": "car", "frame_idx": 101, "time_sec": 10.1},
            {"track_id": 11, "class": "car", "frame_idx": 102, "time_sec": 10.2},
            {"track_id": 11, "class": "car", "frame_idx": 103, "time_sec": 10.3},
        ]
        out = build_label_episode_ranges(
            tracks,
            label="car",
            max_gap_frames=2,
            min_episode_frames=2,
            per_track=False,
        )
        self.assertEqual(len(out), 1)

    def test_resolve_color_track_ids(self) -> None:
        moments = [
            {
                "moment_index": 0,
                "type": "APPEAR",
                "start_time": 1.0,
                "end_time": 1.0,
                "entities": [7],
                "metadata": {"label": "truck", "label_group": "truck", "color_tags": ["white"]},
            },
            {
                "moment_index": 1,
                "type": "APPEAR",
                "start_time": 2.0,
                "end_time": 2.0,
                "entities": [9],
                "metadata": {"label": "truck", "label_group": "truck", "color_tags": ["red"]},
            },
        ]
        white_ids = resolve_color_track_ids(moments, label="truck", color="white")
        self.assertEqual(white_ids, {"7"})
        red_ids = resolve_color_track_ids(moments, label="truck", color="red")
        self.assertEqual(red_ids, {"9"})


if __name__ == "__main__":
    unittest.main()
