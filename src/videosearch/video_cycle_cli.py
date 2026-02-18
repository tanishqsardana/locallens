from __future__ import annotations

import argparse
import json
from pathlib import Path

from .video_cycle import (
    LLMVocabPostprocessConfig,
    VLMCaptionConfig,
    convert_bytetrack_mot_file,
    run_video_cycle,
    video_fps,
)


def _parse_seed_labels(value: str | None) -> list[str]:
    if not value:
        return []
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
        default="List all visible traffic objects in this traffic frame in one short sentence.",
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
        default="car,truck,person",
        help="Comma-separated seed labels for prompt terms",
    )
    parser.add_argument("--target-fps", type=float, default=10.0, help="Video ingest sample FPS")
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.captions and args.auto_captions:
        raise ValueError("Use either --captions or --auto-captions, not both")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_path: str
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
    else:
        raise ValueError("Provide either --tracks or --bytetrack-txt")

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
        output_dir=out_dir,
        captions_path=args.captions,
        vlm_caption_config=vlm_config,
        llm_vocab_postprocess_config=llm_vocab_config,
        synonyms_path=args.synonyms,
        seed_labels=_parse_seed_labels(args.seed_labels),
        target_fps=args.target_fps,
        moment_overrides=_load_moment_overrides(args.moment_config),
        include_full_phase_outputs=bool(args.show_full_phase_outputs),
        phase_preview_limit=int(args.phase_preview_limit),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.show_phase_outputs or args.show_full_phase_outputs:
        phase_path = Path(summary["phase_outputs"])
        phase_payload = json.loads(phase_path.read_text(encoding="utf-8"))
        print(json.dumps({"phase_outputs": phase_payload}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
