from __future__ import annotations

import argparse
import json
from pathlib import Path

from .moment_overlay import OverlayConfig, render_moment_overlay_video


def _resolve(run_dir: str | None, explicit: str | None, filename: str) -> str:
    if explicit:
        return explicit
    if run_dir:
        return str(Path(run_dir) / filename)
    raise ValueError(f"Missing required path for {filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render verification overlay video for moments/tracks")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--run-dir", default=None, help="Directory containing normalized_tracks.json and moments.json")
    parser.add_argument("--tracks", default=None, help="Track JSON path (defaults to run-dir/normalized_tracks.json)")
    parser.add_argument("--moments", default=None, help="Moments JSON path (defaults to run-dir/moments.json)")
    parser.add_argument("--out-video", required=True, help="Output overlay video path")
    parser.add_argument("--show-all-tracks", action="store_true", help="Draw all tracks, not only active moment entities")
    parser.add_argument("--point-tolerance-frames", type=int, default=1)
    parser.add_argument("--max-moment-lines", type=int, default=6)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.5)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--log-every-frames", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tracks_path = _resolve(args.run_dir, args.tracks, "normalized_tracks.json")
    moments_path = _resolve(args.run_dir, args.moments, "moments.json")
    cfg = OverlayConfig(
        codec=str(args.codec),
        show_all_tracks=bool(args.show_all_tracks),
        point_tolerance_frames=int(args.point_tolerance_frames),
        max_moment_lines=int(args.max_moment_lines),
        line_thickness=int(args.line_thickness),
        font_scale=float(args.font_scale),
        start_sec=float(args.start_sec),
        end_sec=float(args.end_sec) if args.end_sec is not None else None,
        log_every_frames=int(args.log_every_frames),
    )
    cfg.validate()
    result = render_moment_overlay_video(
        video_path=args.video,
        tracks_path=tracks_path,
        moments_path=moments_path,
        output_video_path=args.out_video,
        config=cfg,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
