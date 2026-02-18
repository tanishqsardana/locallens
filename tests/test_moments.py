from __future__ import annotations

import unittest

from videosearch.moments import MomentConfig, TrackObservation, generate_moments


def _obs(
    track_id: int,
    label: str,
    frame: int,
    cx: float,
    cy: float,
    *,
    w: float = 10.0,
    h: float = 10.0,
    fps: float = 10.0,
    confidence: float = 0.95,
) -> TrackObservation:
    half_w = w / 2.0
    half_h = h / 2.0
    return TrackObservation(
        track_id=track_id,
        label=label,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=confidence,
        frame_idx=frame,
        time_sec=frame / fps,
    )


class MomentGenerationTest(unittest.TestCase):
    def test_appear_and_disappear(self) -> None:
        observations: list[TrackObservation] = []

        # Anchor track keeps timeline alive through frame 25.
        for frame in range(0, 26):
            observations.append(_obs(99, "tree", frame, 5, 5))

        # Track 1 appears for 5 frames then disappears.
        for frame in range(0, 5):
            observations.append(_obs(1, "car", frame, 20 + frame, 40))

        config = MomentConfig(
            appear_persist_frames=5,
            disappear_missing_frames=10,
            stop_enter_frames=4,
            stop_exit_frames=4,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)

        appear = [m for m in moments if m.type == "APPEAR" and m.entities == [1]]
        disappear = [m for m in moments if m.type == "DISAPPEAR" and m.entities == [1]]

        self.assertEqual(len(appear), 1)
        self.assertAlmostEqual(appear[0].start_time, 0.0, places=3)
        self.assertAlmostEqual(appear[0].end_time, 0.0, places=3)

        self.assertEqual(len(disappear), 1)
        self.assertGreaterEqual(disappear[0].start_time, 1.3)
        self.assertAlmostEqual(disappear[0].end_time, disappear[0].start_time, places=3)

    def test_appear_with_sampled_frame_gaps(self) -> None:
        observations: list[TrackObservation] = []
        for frame in range(0, 26, 2):
            observations.append(_obs(99, "car", frame, 5, 5))
        for frame in [0, 2, 4, 6, 8]:
            observations.append(_obs(1, "car", frame, 20 + frame, 40))

        config = MomentConfig(
            appear_persist_frames=5,
            continuity_max_gap_frames=3,
            disappear_missing_frames=50,
            disappear_min_visible_frames=2,
            stop_enter_frames=4,
            stop_exit_frames=4,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)
        appear = [m for m in moments if m.type == "APPEAR" and m.entities == [1]]
        self.assertEqual(len(appear), 1)

    def test_disappear_requires_min_visible_frames(self) -> None:
        observations: list[TrackObservation] = []
        for frame in range(0, 20):
            observations.append(_obs(99, "car", frame, 10, 10))
        for frame in [0, 1]:
            observations.append(_obs(2, "car", frame, 30 + frame, 10))

        config = MomentConfig(
            appear_persist_frames=2,
            disappear_missing_frames=4,
            disappear_min_visible_frames=5,
            stop_enter_frames=4,
            stop_exit_frames=4,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)
        disappear = [m for m in moments if m.type == "DISAPPEAR" and m.entities == [2]]
        self.assertEqual(disappear, [])

    def test_stop_event_from_speed_transition(self) -> None:
        observations: list[TrackObservation] = []
        # Anchor track.
        for frame in range(0, 40):
            observations.append(_obs(99, "road", frame, 3, 3))

        # Track 2 moves, stops, then moves again.
        for frame in range(0, 8):
            observations.append(_obs(2, "car", frame, 10 + 2 * frame, 50))
        for frame in range(8, 20):
            observations.append(_obs(2, "car", frame, 24, 50))
        for frame in range(20, 36):
            observations.append(_obs(2, "car", frame, 24 + 2 * (frame - 20), 50))

        config = MomentConfig(
            appear_persist_frames=3,
            disappear_missing_frames=12,
            stop_enter_frames=8,
            stop_exit_frames=8,
            stop_speed_threshold=0.015,
            movement_speed_threshold=0.04,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)

        stop_moments = [m for m in moments if m.type == "STOP" and m.entities == [2]]
        self.assertEqual(len(stop_moments), 1)
        stop = stop_moments[0]
        self.assertLess(stop.start_time, stop.end_time)
        self.assertGreaterEqual(stop.start_time, 0.7)
        self.assertGreaterEqual(stop.end_time, 2.0)

    def test_near_moments_are_merged_when_gap_is_small(self) -> None:
        observations: list[TrackObservation] = []
        for frame in range(0, 30):
            # Car at center.
            observations.append(_obs(10, "car", frame, 50, 50))
            # Person near/far pattern to create two near intervals with a tiny gap.
            if 2 <= frame <= 6 or 9 <= frame <= 13:
                observations.append(_obs(11, "person", frame, 53, 50))
            else:
                observations.append(_obs(11, "person", frame, 85, 50))

        config = MomentConfig(
            appear_persist_frames=2,
            disappear_missing_frames=10,
            stop_enter_frames=8,
            stop_exit_frames=8,
            near_enter_frames=2,
            near_exit_frames=2,
            near_threshold=0.04,
            near_threshold_exit=0.12,
            approach_window=8,
            approach_reverse_frames=8,
            merge_gap_sec=1.0,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)
        near = [m for m in moments if m.type == "NEAR" and m.entities == [10, 11]]
        self.assertEqual(len(near), 1)
        self.assertLessEqual(near[0].start_time, 0.2)
        self.assertGreaterEqual(near[0].end_time, 1.4)

    def test_approach_event_from_decreasing_distance(self) -> None:
        observations: list[TrackObservation] = []
        for frame in range(0, 35):
            observations.append(_obs(20, "car", frame, 80, 60))
            # Person approaches for 10 frames, then moves away for 10+ frames.
            if frame <= 10:
                px = 10 + 5 * frame
            else:
                px = 60 - 4 * (frame - 10)
            observations.append(_obs(21, "person", frame, px, 60))

        config = MomentConfig(
            appear_persist_frames=2,
            disappear_missing_frames=15,
            stop_enter_frames=8,
            stop_exit_frames=8,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
            approach_drop_threshold=0.08,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)
        approach = [m for m in moments if m.type == "APPROACH" and m.entities == [20, 21]]
        self.assertEqual(len(approach), 1)
        self.assertLess(approach[0].start_time, approach[0].end_time)
        self.assertGreater(approach[0].metadata.get("distance_drop", 0.0), 0.08)

    def test_traffic_change_event_from_object_count_shift(self) -> None:
        observations: list[TrackObservation] = []
        for frame in range(0, 10):
            observations.append(_obs(1, "car", frame, 20, 40))
        for frame in range(10, 20):
            observations.append(_obs(1, "car", frame, 20, 40))
            observations.append(_obs(2, "car", frame, 30, 40))
            observations.append(_obs(3, "car", frame, 40, 40))
            observations.append(_obs(4, "car", frame, 50, 40))
        for frame in range(20, 30):
            observations.append(_obs(1, "car", frame, 20, 40))

        config = MomentConfig(
            appear_persist_frames=2,
            disappear_missing_frames=100,
            stop_enter_frames=8,
            stop_exit_frames=8,
            near_enter_frames=3,
            near_exit_frames=3,
            approach_window=8,
            approach_reverse_frames=8,
            emit_traffic_change=True,
            traffic_change_window_frames=5,
            traffic_change_threshold=2,
            traffic_change_cooldown_frames=4,
        )
        moments = generate_moments(observations, frame_width=100, frame_height=100, config=config)
        traffic = [m for m in moments if m.type == "TRAFFIC_CHANGE"]
        self.assertGreaterEqual(len(traffic), 1)


if __name__ == "__main__":
    unittest.main()
