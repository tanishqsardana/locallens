from __future__ import annotations

import unittest

import numpy as np

from videosearch.video_cycle import (
    DetectionTrackingConfig,
    TrackProcessingConfig,
    VideoManifest,
    YOLOWorldConfig,
    build_moment_label_allowlist,
    build_phase_outputs,
    build_keyframe_targets,
    build_groundingdino_caption,
    build_prompt_terms,
    build_vocab_postprocess_prompt,
    convert_bytetrack_mot_rows,
    extract_chat_completion_text,
    extract_object_nouns,
    normalize_detection_rows,
    normalize_tracking_rows,
    parse_vocab_postprocess_output,
    process_tracking_rows,
    track_detections,
)


class VideoCycleHelpersTest(unittest.TestCase):
    def test_yoloworld_config_validate(self) -> None:
        cfg = YOLOWorldConfig()
        cfg.validate()

    def test_extract_object_nouns(self) -> None:
        captions = [
            "A red cars waits near the crosswalk.",
            "Two cars and one truck move through the lane.",
            "A pedestrian crosses while the car slows down.",
        ]
        nouns = extract_object_nouns(captions, min_count=1, top_k=20)
        self.assertIn("car", nouns)
        self.assertIn("truck", nouns)
        self.assertIn("crosswalk", nouns)

    def test_prompt_terms_with_synonyms(self) -> None:
        seed = ["automobile", "Pedestrian"]
        discovered = ["cars", "pickup truck", "walker"]
        synonyms = {"automobile": "car", "walker": "person", "cars": "car"}
        terms = build_prompt_terms(seed, discovered, synonyms)
        self.assertIn("car", terms)
        self.assertIn("person", terms)
        self.assertIn("pickup truck", terms)

    def test_build_moment_label_allowlist(self) -> None:
        allow = build_moment_label_allowlist(["cars", "truck", "road"], {"cars": "car"})
        self.assertEqual(allow, ["car", "truck", "road"])

    def test_build_vocab_postprocess_prompt(self) -> None:
        prompt = build_vocab_postprocess_prompt(
            seed_labels=["car", "truck", "person"],
            discovered_labels=["car", "cloudy", "lane", "billboard"],
            prompt_terms=["car", "cloudy", "lane"],
            max_detection_terms=12,
        )
        self.assertIn("Seed labels", prompt)
        self.assertIn("Current prompt terms", prompt)
        self.assertIn("12", prompt)

    def test_build_groundingdino_caption(self) -> None:
        caption = build_groundingdino_caption(["car", " truck ", "person"])
        self.assertEqual(caption, "car . truck . person .")

    def test_normalize_tracking_rows_with_flexible_keys(self) -> None:
        rows = [
            {
                "id": 7,
                "label": "automobile",
                "xyxy": [10, 20, 30, 40],
                "score": 0.91,
                "frame": 15,
                "timestamp": 0.5,
            },
            {
                "track_id": 9,
                "class": "truck",
                "bbox": [11, 21, 31, 41],
                "confidence": 0.88,
                "frame_idx": 16,
                "time_sec": 0.6,
            },
        ]
        normalized = normalize_tracking_rows(
            rows,
            synonym_map={"automobile": "car"},
            allowed_labels={"car", "truck"},
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["class"], "car")
        self.assertEqual(normalized[1]["class"], "truck")

    def test_normalize_detection_rows(self) -> None:
        rows = [
            {
                "label": "automobiles",
                "xyxy": [10, 20, 30, 40],
                "score": 0.91,
                "frame": 15,
                "timestamp": 0.5,
            },
            {
                "class": "truck",
                "bbox": [11, 21, 31, 41],
                "confidence": 0.88,
                "frame_idx": 16,
                "time_sec": 0.6,
            },
        ]
        normalized = normalize_detection_rows(
            rows,
            synonym_map={"automobile": "car"},
            allowed_labels={"car", "truck"},
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["class"], "car")
        self.assertEqual(normalized[1]["class"], "truck")

    def test_track_detections_assigns_ids(self) -> None:
        detections = [
            {"class": "car", "bbox": [10, 10, 20, 20], "confidence": 0.9, "frame_idx": 0, "time_sec": 0.0},
            {"class": "car", "bbox": [11, 10, 21, 20], "confidence": 0.9, "frame_idx": 1, "time_sec": 0.1},
            {"class": "truck", "bbox": [50, 50, 70, 70], "confidence": 0.95, "frame_idx": 1, "time_sec": 0.1},
            {"class": "car", "bbox": [12, 10, 22, 20], "confidence": 0.9, "frame_idx": 2, "time_sec": 0.2},
        ]
        tracked, report = track_detections(
            detections,
            config=DetectionTrackingConfig(
                iou_threshold=0.3,
                max_missed_frames=5,
                min_detection_confidence=0.0,
                class_aware=True,
            ),
            fps=10.0,
        )
        self.assertEqual(len(tracked), 4)
        self.assertGreaterEqual(report["track_count"], 2)
        car_track_ids = [row["track_id"] for row in tracked if row["class"] == "car"]
        self.assertEqual(len(set(car_track_ids)), 1)

    def test_process_tracking_rows_filters_and_interpolates(self) -> None:
        rows = [
            {"track_id": 1, "class": "car", "bbox": [10, 20, 30, 40], "confidence": 0.9, "frame_idx": 0, "time_sec": 0.0},
            {"track_id": 1, "class": "car", "bbox": [12, 20, 32, 40], "confidence": 0.9, "frame_idx": 2, "time_sec": 0.2},
            {"track_id": 2, "class": "car", "bbox": [50, 50, 60, 60], "confidence": 0.2, "frame_idx": 5, "time_sec": 0.5},
            {"track_id": 3, "class": "truck", "bbox": [70, 70, 80, 80], "confidence": 0.95, "frame_idx": 10, "time_sec": 1.0},
        ]
        processed, report = process_tracking_rows(
            rows,
            config=TrackProcessingConfig(
                min_confidence=0.5,
                min_track_length_frames=2,
                max_interp_gap_frames=1,
                clip_bboxes_to_frame=True,
            ),
            frame_width=100,
            frame_height=100,
            fps=10.0,
        )
        self.assertGreaterEqual(len(processed), 3)  # includes one interpolated frame
        self.assertEqual(report["removed_low_confidence_count"], 1)
        self.assertEqual(report["removed_short_track_count"], 1)
        self.assertGreaterEqual(report["interpolated_rows_added_count"], 1)

    def test_build_keyframe_targets(self) -> None:
        moments = [
            {"start_time": 1.0, "end_time": 2.0},
            {"start_time": 3.5, "end_time": 3.5},
        ]
        targets = build_keyframe_targets(moments, fps=10.0, frame_count=200)
        self.assertEqual(len(targets), 2)
        first = targets[0]["frames"]
        self.assertTrue(any(item["role"] == "start" for item in first))
        self.assertTrue(any(item["role"] == "middle" for item in first))
        self.assertTrue(any(item["role"] == "end" for item in first))

        second = targets[1]["frames"]
        # Point event can collapse to one frame after de-duplication.
        self.assertGreaterEqual(len(second), 1)

    def test_build_phase_outputs_preview_and_full(self) -> None:
        manifest = VideoManifest(
            video_id="vid-1",
            video_path="/tmp/vid-1.mp4",
            width=1920,
            height=1080,
            fps=10.0,
            frame_count=100,
            duration_sec=10.0,
            target_fps=10.0,
            sampled_frame_count=3,
            sampled_frames_path="/tmp/sampled_frames.json",
        )
        sampled_rows = [
            {"frame_idx": 0, "time_sec": 0.0, "image_path": "a.jpg"},
            {"frame_idx": 10, "time_sec": 1.0, "image_path": "b.jpg"},
        ]
        raw_tracks = [
            {"track_id": 1, "class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9, "frame_idx": 0, "time_sec": 0.0},
            {"track_id": 1, "class": "car", "bbox": [1, 0, 11, 10], "confidence": 0.9, "frame_idx": 1, "time_sec": 0.1},
        ]
        normalized_tracks = list(raw_tracks)
        moment_rows = [
            {
                "moment_index": 0,
                "video_id": "vid-1",
                "type": "APPEAR",
                "start_time": 0.0,
                "end_time": 0.0,
                "duration_sec": 0.0,
                "entities": [1],
                "metadata": {},
            },
            {
                "moment_index": 1,
                "video_id": "vid-1",
                "type": "STOP",
                "start_time": 1.0,
                "end_time": 2.0,
                "duration_sec": 1.0,
                "entities": [1],
                "metadata": {},
            },
        ]
        keyframe_rows = [
            {"moment_index": 0, "role": "start", "frame_idx": 0, "time_sec": 0.0, "image_path": "k0.jpg"},
            {"moment_index": 1, "role": "middle", "frame_idx": 15, "time_sec": 1.5, "image_path": "k1.jpg"},
        ]
        embeddings = {0: np.ones(12, dtype=np.float32), 1: np.arange(12, dtype=np.float32)}

        preview = build_phase_outputs(
            manifest=manifest,
            sampled_rows=sampled_rows,
            raw_tracks=raw_tracks,
            normalized_tracks=normalized_tracks,
            moment_rows=moment_rows,
            keyframe_rows=keyframe_rows,
            embeddings=embeddings,
            embedding_model_name="hist",
            index_db_path="/tmp/does_not_exist.sqlite",
            synonym_map={"automobile": "car"},
            captions=["a car drives", "the car stops"],
            caption_rows=[{"frame_idx": 0, "caption": "a car drives"}],
            discovered_labels=["car"],
            prompt_terms=["car"],
            moment_label_allowlist=["car", "truck"],
            phase2_status="provided_captions",
            include_full=False,
            preview_limit=1,
        )
        self.assertIn("phase_1_ingest", preview)
        self.assertEqual(len(preview["phase_3_normalized_tracks"]["rows"]), 1)
        self.assertEqual(preview["phase_4_moments"]["moment_type_counts"]["APPEAR"], 1)

        full = build_phase_outputs(
            manifest=manifest,
            sampled_rows=sampled_rows,
            raw_tracks=raw_tracks,
            normalized_tracks=normalized_tracks,
            moment_rows=moment_rows,
            keyframe_rows=keyframe_rows,
            embeddings=embeddings,
            embedding_model_name="hist",
            index_db_path="/tmp/does_not_exist.sqlite",
            synonym_map={"automobile": "car"},
            captions=["a car drives", "the car stops"],
            caption_rows=[{"frame_idx": 0, "caption": "a car drives"}],
            discovered_labels=["car"],
            prompt_terms=["car"],
            moment_label_allowlist=["car", "truck"],
            phase2_status="provided_captions",
            include_full=True,
            preview_limit=1,
        )
        self.assertEqual(len(full["phase_3_normalized_tracks"]["rows"]), len(normalized_tracks))
        self.assertEqual(len(full["phase_6_embeddings"]["rows"]), len(embeddings))
        self.assertIn("llm_postprocess", full["phase_2_vocabulary"])

    def test_convert_bytetrack_mot_rows(self) -> None:
        mot_rows = [
            "1,2,100,200,50,80,0.91,-1,-1,-1",
            "2,2,102,201,50,80,0.89,-1,-1,-1",
            "2,7,300,220,60,90,0.20,-1,-1,-1",
        ]
        converted = convert_bytetrack_mot_rows(
            mot_rows,
            fps=10.0,
            class_label="car",
            min_score=0.5,
        )
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["frame_idx"], 0)
        self.assertAlmostEqual(converted[0]["time_sec"], 0.0, places=3)
        self.assertEqual(converted[0]["bbox"], [100.0, 200.0, 150.0, 280.0])
        self.assertEqual(converted[1]["frame_idx"], 1)
        self.assertEqual(converted[1]["class"], "car")

    def test_extract_chat_completion_text(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "The frame shows a red car and a white SUV."
                    }
                }
            ]
        }
        text = extract_chat_completion_text(response)
        self.assertEqual(text, "The frame shows a red car and a white SUV.")

    def test_parse_vocab_postprocess_output(self) -> None:
        raw = """
```json
{
  "detection_terms": ["car", "truck", "car", "billboard"],
  "scene_terms": ["highway", "cloudy sky"],
  "dropped_terms": ["visible", "show"],
  "canonical_map": {"cars": "car", "trucks": "truck"}
}
```
"""
        parsed = parse_vocab_postprocess_output(raw, max_detection_terms=3)
        self.assertEqual(parsed["detection_terms"], ["car", "truck", "billboard"])
        self.assertIn("highway", parsed["scene_terms"])
        self.assertEqual(parsed["canonical_map"]["cars"], "car")


if __name__ == "__main__":
    unittest.main()
