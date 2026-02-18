from __future__ import annotations

import unittest

from videosearch.moment_query import (
    PassThroughConfig,
    answer_nlq,
    appearance_episodes,
    frames_with_label,
    pass_through_tracks,
    when_object_appears,
)


class MomentQueryTest(unittest.TestCase):
    def test_when_object_appears(self) -> None:
        moments = [
            {
                "moment_index": 0,
                "type": "APPEAR",
                "start_time": 1.0,
                "end_time": 1.0,
                "entities": [7],
                "metadata": {"label": "truck", "label_group": "truck"},
            },
            {
                "moment_index": 1,
                "type": "APPEAR",
                "start_time": 2.0,
                "end_time": 2.0,
                "entities": [8],
                "metadata": {"label": "car", "label_group": "car"},
            },
        ]
        out = when_object_appears(moments, label="truck")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["time_sec"], 1.0)
        self.assertEqual(out[0]["entities"], [7])

    def test_frames_with_label(self) -> None:
        tracks = [
            {"track_id": 1, "class": "truck", "frame_idx": 10, "time_sec": 1.0},
            {"track_id": 2, "class": "truck", "frame_idx": 11, "time_sec": 1.1},
            {"track_id": 3, "class": "truck", "frame_idx": 15, "time_sec": 1.5},
            {"track_id": 4, "class": "car", "frame_idx": 10, "time_sec": 1.0},
        ]
        out = frames_with_label(tracks, label="truck", max_gap_frames=2)
        self.assertEqual(len(out["frames"]), 3)
        self.assertGreaterEqual(len(out["intervals"]), 2)

    def test_appearance_episodes(self) -> None:
        tracks = [
            {"track_id": 1, "class": "truck", "frame_idx": 10, "time_sec": 1.0},
            {"track_id": 1, "class": "truck", "frame_idx": 11, "time_sec": 1.1},
            {"track_id": 2, "class": "truck", "frame_idx": 20, "time_sec": 2.0},
            {"track_id": 2, "class": "truck", "frame_idx": 21, "time_sec": 2.1},
        ]
        out = appearance_episodes(tracks, label="truck", max_gap_frames=2, min_episode_frames=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["start_frame"], 10)
        self.assertEqual(out[1]["start_frame"], 20)

    def test_pass_through_tracks(self) -> None:
        tracks = [
            {"track_id": 10, "class": "car", "bbox": [0, 40, 10, 50], "frame_idx": 0, "time_sec": 0.0},
            {"track_id": 10, "class": "car", "bbox": [10, 40, 20, 50], "frame_idx": 1, "time_sec": 0.5},
            {"track_id": 10, "class": "car", "bbox": [30, 40, 40, 50], "frame_idx": 2, "time_sec": 1.0},
            {"track_id": 10, "class": "car", "bbox": [60, 40, 70, 50], "frame_idx": 3, "time_sec": 1.5},
            {"track_id": 10, "class": "car", "bbox": [90, 40, 100, 50], "frame_idx": 4, "time_sec": 2.0},
        ]
        out = pass_through_tracks(
            tracks,
            label="car",
            frame_width=100,
            frame_height=100,
            config=PassThroughConfig(
                min_track_frames=3,
                min_duration_sec=1.0,
                border_margin_ratio=0.1,
                min_displacement_norm=0.1,
            ),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 10)
        self.assertTrue(out[0]["entered_border"])
        self.assertTrue(out[0]["exited_border"])

    def test_answer_nlq(self) -> None:
        moments = [
            {
                "moment_index": 0,
                "type": "APPEAR",
                "start_time": 3.0,
                "end_time": 3.0,
                "entities": [2],
                "metadata": {"label": "truck", "label_group": "truck"},
            }
        ]
        tracks = [
            {"track_id": 2, "class": "truck", "bbox": [10, 10, 20, 20], "frame_idx": 30, "time_sec": 3.0}
        ]
        out = answer_nlq(
            "when does truck appear?",
            moments=moments,
            tracks=tracks,
            frame_width=100,
            frame_height=100,
        )
        self.assertEqual(out["intent"], "appear")
        self.assertEqual(len(out["results"]), 1)
        self.assertIn("episodes", out)


if __name__ == "__main__":
    unittest.main()
