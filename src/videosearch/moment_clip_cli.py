from __future__ import annotations

import argparse
import json
from pathlib import Path

from .moment_clip import ClipExportConfig, export_label_episode_clips


def _resolve(run_dir: str | None, explicit: str | None, filename: str) -> str:
    if explicit:
        return explicit
    if run_dir:
        return str(Path(run_dir) / filename)
    raise ValueError(f"Missing required path for {filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-label appearance episode clips with optional bbox overlays")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--label", required=True, help="Target label (e.g., car, truck)")
    parser.add_argument("--run-dir", default=None, help="Directory containing normalized_tracks.json")
    parser.add_argument("--tracks", default=None, help="Track JSON path (defaults to run-dir/normalized_tracks.json)")
    parser.add_argument("--out-dir", required=True, help="Output directory for MP4 clips and summary JSON")
    parser.add_argument("--padding-sec", type=float, default=0.3, help="Seconds to pad before/after each episode")
    parser.add_argument("--max-gap-frames", type=int, default=2, help="Max frame gap for merging episode frames")
    parser.add_argument("--min-episode-frames", type=int, default=2, help="Minimum frames for an episode")
    parser.add_argument("--no-overlay-boxes", action="store_true", help="Disable bbox overlays in exported clips")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.5)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--log-every-frames", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tracks_path = _resolve(args.run_dir, args.tracks, "normalized_tracks.json")
    cfg = ClipExportConfig(
        codec=str(args.codec),
        padding_sec=float(args.padding_sec),
        overlay_boxes=not bool(args.no_overlay_boxes),
        line_thickness=int(args.line_thickness),
        font_scale=float(args.font_scale),
        max_gap_frames=int(args.max_gap_frames),
        min_episode_frames=int(args.min_episode_frames),
        log_every_frames=int(args.log_every_frames),
    )
    cfg.validate()
    result = export_label_episode_clips(
        video_path=args.video,
        tracks_path=tracks_path,
        label=args.label,
        output_dir=args.out_dir,
        config=cfg,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
