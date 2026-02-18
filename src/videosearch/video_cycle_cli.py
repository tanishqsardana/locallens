from __future__ import annotations

import argparse
import json
from pathlib import Path

from .video_cycle import convert_bytetrack_mot_file, run_video_cycle, video_fps


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

    summary = run_video_cycle(
        video_path=args.video,
        tracks_path=tracks_path,
        output_dir=out_dir,
        captions_path=args.captions,
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
