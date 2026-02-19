from __future__ import annotations

import argparse
import json
from pathlib import Path

from .video_cycle import (
    DetectionTrackingConfig,
    GroundingDINOConfig,
    YOLOWorldConfig,
    LLMVocabPostprocessConfig,
    SCENE_PROFILE_AUTO,
    SCENE_PROFILE_PEDESTRIAN,
    SCENE_PROFILE_TRAFFIC,
    TrackProcessingConfig,
    VLMCaptionConfig,
    convert_bytetrack_mot_file,
    run_video_cycle,
    video_fps,
)


def _parse_seed_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_label_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_moment_overrides(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Moment override file must be a JSON object")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end video moment generation cycle")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--tracks", default=None, help="Tracking output JSON path")
    parser.add_argument("--bytetrack-txt", default=None, help="ByteTrack MOT txt path")
    parser.add_argument(
        "--detections",
        default=None,
        help="Frame-level detections JSON path (class,bbox,confidence,frame_idx,time_sec). "
        "If provided, tracking is run in-pipeline.",
    )
    parser.add_argument(
        "--auto-detections-groundingdino",
        action="store_true",
        help="Run GroundingDINO on sampled frames inside pipeline and use those detections for tracking.",
    )
    parser.add_argument(
        "--auto-detections-yoloworld",
        action="store_true",
        help="Run YOLO-World on sampled frames inside pipeline and use those detections for tracking.",
    )
    parser.add_argument("--groundingdino-config-path", default=None, help="GroundingDINO model config file path")
    parser.add_argument("--groundingdino-weights-path", default=None, help="GroundingDINO model weights file path")
    parser.add_argument("--groundingdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-text-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-device", default="cuda")
    parser.add_argument("--groundingdino-frame-stride", type=int, default=1)
    parser.add_argument("--groundingdino-max-frames", type=int, default=0)
    parser.add_argument("--yoloworld-model", default="yolov8s-worldv2.pt")
    parser.add_argument("--yoloworld-confidence", type=float, default=0.2)
    parser.add_argument("--yoloworld-iou-threshold", type=float, default=0.7)
    parser.add_argument("--yoloworld-device", default="cuda")
    parser.add_argument("--yoloworld-frame-stride", type=int, default=1)
    parser.add_argument("--yoloworld-max-frames", type=int, default=0)
    parser.add_argument(
        "--bytetrack-class",
        default="car",
        help="Class label to assign to converted ByteTrack tracks",
    )
    parser.add_argument(
        "--bytetrack-min-score",
        type=float,
        default=0.0,
        help="Minimum MOT score to keep during ByteTrack conversion",
    )
    parser.add_argument("--out-dir", default="data/video_cycle", help="Output directory")
    parser.add_argument("--captions", default=None, help="Optional VLM captions JSON path")
    parser.add_argument(
        "--auto-captions",
        action="store_true",
        help="Generate captions from sampled frames via OpenAI-compatible VLM endpoint",
    )
    parser.add_argument(
        "--vlm-endpoint",
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model name served by the endpoint",
    )
    parser.add_argument(
        "--vlm-prompt",
        default=(
            "List visible objects and scene elements as a comma-separated list of singular nouns, "
            "lowercase, no adjectives/colors/verbs/locations, no duplicates, max 20 terms."
        ),
        help="Prompt sent with each frame",
    )
    parser.add_argument("--vlm-max-tokens", type=int, default=120, help="Max caption tokens")
    parser.add_argument("--vlm-frame-stride", type=int, default=10, help="Caption every N sampled frames")
    parser.add_argument("--vlm-timeout-sec", type=float, default=60.0, help="VLM request timeout")
    parser.add_argument("--vlm-temperature", type=float, default=0.0, help="VLM sampling temperature")
    parser.add_argument("--vlm-api-key", default=None, help="Optional bearer token for VLM endpoint")
    parser.add_argument(
        "--llm-postprocess-vocab",
        action="store_true",
        help="Post-process discovered vocabulary using an LLM and use returned detection_terms",
    )
    parser.add_argument(
        "--llm-postprocess-endpoint",
        default=None,
        help="Endpoint for vocab postprocess LLM (defaults to --vlm-endpoint)",
    )
    parser.add_argument(
        "--llm-postprocess-model",
        default=None,
        help="Model for vocab postprocess LLM (defaults to --vlm-model)",
    )
    parser.add_argument("--llm-postprocess-max-tokens", type=int, default=500)
    parser.add_argument("--llm-postprocess-timeout-sec", type=float, default=60.0)
    parser.add_argument("--llm-postprocess-temperature", type=float, default=0.0)
    parser.add_argument("--llm-postprocess-api-key", default=None)
    parser.add_argument("--llm-postprocess-max-detection-terms", type=int, default=20)
    parser.add_argument("--synonyms", default=None, help="Optional synonym map JSON path")
    parser.add_argument(
        "--seed-labels",
        default=None,
        help="Optional comma-separated seed labels for prompt terms (defaults by scene profile)",
    )
    parser.add_argument(
        "--moment-labels",
        default=None,
        help="Optional comma-separated labels allowed into detection/tracking/moment phases (defaults by scene profile)",
    )
    parser.add_argument(
        "--scene-profile",
        default=SCENE_PROFILE_AUTO,
        choices=[SCENE_PROFILE_AUTO, SCENE_PROFILE_TRAFFIC, SCENE_PROFILE_PEDESTRIAN],
        help="Scene profile for defaults/moment behavior",
    )
    parser.add_argument("--target-fps", type=float, default=10.0, help="Video ingest sample FPS")
    parser.add_argument("--detect-track-iou-threshold", type=float, default=0.3)
    parser.add_argument("--detect-track-max-missed-frames", type=int, default=10)
    parser.add_argument("--detect-track-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--detect-track-class-agnostic",
        action="store_true",
        help="Disable class-aware matching when tracking detections in-pipeline",
    )
    parser.add_argument("--track-min-confidence", type=float, default=0.0)
    parser.add_argument("--track-min-length", type=int, default=1, help="Minimum detections per track to keep")
    parser.add_argument(
        "--track-max-interp-gap",
        type=int,
        default=0,
        help="Interpolate missing detections for gaps up to this many frames",
    )
    parser.add_argument(
        "--track-no-clip-bbox",
        action="store_true",
        help="Disable clipping bounding boxes to frame bounds",
    )
    parser.add_argument(
        "--moment-config",
        default=None,
        help="Optional JSON file with MomentConfig overrides",
    )
    parser.add_argument(
        "--show-phase-outputs",
        action="store_true",
        help="Print consolidated phase outputs after the run",
    )
    parser.add_argument(
        "--show-full-phase-outputs",
        action="store_true",
        help="Persist and print untruncated phase outputs",
    )
    parser.add_argument(
        "--phase-preview-limit",
        type=int,
        default=25,
        help="Rows per phase to keep when not using --show-full-phase-outputs",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Print phase progress logs to stderr during long-running steps",
    )
    parser.add_argument(
        "--disable-semantic-index",
        action="store_true",
        help="Disable text-semantic indexing of moments in Phase 6/7",
    )
    parser.add_argument(
        "--semantic-index-embedder",
        default="hashing",
        choices=["hashing", "sentence-transformer"],
        help="Embedder for text-semantic moment indexing",
    )
    parser.add_argument(
        "--semantic-index-model",
        default=None,
        help="Optional model name for semantic index embedder",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.captions and args.auto_captions:
        raise ValueError("Use either --captions or --auto-captions, not both")
    input_count = (
        int(bool(args.tracks))
        + int(bool(args.bytetrack_txt))
        + int(bool(args.detections))
        + int(bool(args.auto_detections_groundingdino))
        + int(bool(args.auto_detections_yoloworld))
    )
    if input_count != 1:
        raise ValueError(
            "Provide exactly one of --tracks, --bytetrack-txt, --detections, "
            "--auto-detections-groundingdino, or --auto-detections-yoloworld"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_path: str | None = None
    detections_path: str | None = None
    groundingdino_config: GroundingDINOConfig | None = None
    yoloworld_config: YOLOWorldConfig | None = None
    if args.tracks:
        tracks_path = args.tracks
    elif args.bytetrack_txt:
        fps = video_fps(args.video)
        converted = convert_bytetrack_mot_file(
            mot_txt_path=args.bytetrack_txt,
            fps=fps,
            class_label=args.bytetrack_class,
            min_score=float(args.bytetrack_min_score),
        )
        inputs_dir = out_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        converted_path = inputs_dir / "converted_tracks_from_bytetrack.json"
        converted_path.write_text(json.dumps(converted, indent=2, ensure_ascii=True), encoding="utf-8")
        tracks_path = str(converted_path)
    elif args.detections:
        detections_path = args.detections
    elif args.auto_detections_groundingdino:
        if not args.groundingdino_config_path or not args.groundingdino_weights_path:
            raise ValueError(
                "--auto-detections-groundingdino requires --groundingdino-config-path and --groundingdino-weights-path"
            )
        groundingdino_config = GroundingDINOConfig(
            model_config_path=str(args.groundingdino_config_path),
            model_weights_path=str(args.groundingdino_weights_path),
            box_threshold=float(args.groundingdino_box_threshold),
            text_threshold=float(args.groundingdino_text_threshold),
            device=str(args.groundingdino_device),
            frame_stride=int(args.groundingdino_frame_stride),
            max_frames=int(args.groundingdino_max_frames),
        )
        groundingdino_config.validate()
    elif args.auto_detections_yoloworld:
        yoloworld_config = YOLOWorldConfig(
            model=str(args.yoloworld_model),
            confidence=float(args.yoloworld_confidence),
            iou_threshold=float(args.yoloworld_iou_threshold),
            device=str(args.yoloworld_device),
            frame_stride=int(args.yoloworld_frame_stride),
            max_frames=int(args.yoloworld_max_frames),
        )
        yoloworld_config.validate()
    else:
        raise ValueError("Provide exactly one input source")

    vlm_config = None
    if args.auto_captions:
        vlm_config = VLMCaptionConfig(
            endpoint=args.vlm_endpoint,
            model=args.vlm_model,
            prompt=args.vlm_prompt,
            max_tokens=int(args.vlm_max_tokens),
            frame_stride=int(args.vlm_frame_stride),
            timeout_sec=float(args.vlm_timeout_sec),
            temperature=float(args.vlm_temperature),
            api_key=args.vlm_api_key,
        )
        vlm_config.validate()

    track_config = TrackProcessingConfig(
        min_confidence=float(args.track_min_confidence),
        min_track_length_frames=int(args.track_min_length),
        max_interp_gap_frames=int(args.track_max_interp_gap),
        clip_bboxes_to_frame=not bool(args.track_no_clip_bbox),
    )
    track_config.validate()

    detection_tracking_config = DetectionTrackingConfig(
        iou_threshold=float(args.detect_track_iou_threshold),
        max_missed_frames=int(args.detect_track_max_missed_frames),
        min_detection_confidence=float(args.detect_track_min_confidence),
        class_aware=not bool(args.detect_track_class_agnostic),
    )
    detection_tracking_config.validate()

    llm_vocab_config = None
    if args.llm_postprocess_vocab:
        llm_vocab_config = LLMVocabPostprocessConfig(
            endpoint=args.llm_postprocess_endpoint or args.vlm_endpoint,
            model=args.llm_postprocess_model or args.vlm_model,
            max_tokens=int(args.llm_postprocess_max_tokens),
            timeout_sec=float(args.llm_postprocess_timeout_sec),
            temperature=float(args.llm_postprocess_temperature),
            api_key=args.llm_postprocess_api_key or args.vlm_api_key,
            max_detection_terms=int(args.llm_postprocess_max_detection_terms),
        )
        llm_vocab_config.validate()

    summary = run_video_cycle(
        video_path=args.video,
        tracks_path=tracks_path,
        detections_path=detections_path,
        groundingdino_config=groundingdino_config,
        yoloworld_config=yoloworld_config,
        output_dir=out_dir,
        detection_tracking_config=(
            detection_tracking_config
            if (detections_path or groundingdino_config or yoloworld_config)
            else None
        ),
        captions_path=args.captions,
        vlm_caption_config=vlm_config,
        llm_vocab_postprocess_config=llm_vocab_config,
        log_progress=bool(args.log_progress),
        synonyms_path=args.synonyms,
        seed_labels=_parse_seed_labels(args.seed_labels),
        moment_label_allowlist=_parse_label_list(args.moment_labels),
        scene_profile=str(args.scene_profile),
        target_fps=args.target_fps,
        moment_overrides=_load_moment_overrides(args.moment_config),
        track_processing_config=track_config,
        include_full_phase_outputs=bool(args.show_full_phase_outputs),
        phase_preview_limit=int(args.phase_preview_limit),
        semantic_index_embedder=str(args.semantic_index_embedder),
        semantic_index_model=args.semantic_index_model,
        enable_semantic_index=not bool(args.disable_semantic_index),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.show_phase_outputs or args.show_full_phase_outputs:
        phase_path = Path(summary["phase_outputs"])
        phase_payload = json.loads(phase_path.read_text(encoding="utf-8"))
        print(json.dumps({"phase_outputs": phase_payload}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
