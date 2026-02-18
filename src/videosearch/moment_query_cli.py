from __future__ import annotations

import argparse
import json
from pathlib import Path

from .moment_query import (
    PassThroughConfig,
    answer_nlq,
    appearance_episodes,
    frames_with_label,
    load_rows,
    pass_through_tracks,
    when_object_appears,
)


def _resolve(default_run_dir: str | None, explicit: str | None, filename: str) -> str:
    if explicit:
        return explicit
    if default_run_dir:
        return str(Path(default_run_dir) / filename)
    raise ValueError(f"Missing required path for {filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query video-cycle moments/tracks for operational questions")
    parser.add_argument("--run-dir", default=None, help="Directory containing moments.json and normalized_tracks.json")

    sub = parser.add_subparsers(dest="command", required=True)

    appear = sub.add_parser("appear", help="When does a label appear?")
    appear.add_argument("--label", required=True)
    appear.add_argument("--moments", default=None)
    appear.add_argument("--tracks", default=None)
    appear.add_argument("--max-gap-frames", type=int, default=2)
    appear.add_argument("--min-episode-frames", type=int, default=2)
    appear.add_argument(
        "--label-union-episodes",
        action="store_true",
        help="Merge all tracks of the label into shared presence windows (legacy behavior)",
    )

    frames = sub.add_parser("frames-with", help="Which frames contain a label?")
    frames.add_argument("--label", required=True)
    frames.add_argument("--tracks", default=None)
    frames.add_argument("--max-gap-frames", type=int, default=2)

    passthrough = sub.add_parser("pass-through", help="When does a label pass through the scene?")
    passthrough.add_argument("--label", required=True)
    passthrough.add_argument("--tracks", default=None)
    passthrough.add_argument("--frame-width", type=int, required=True)
    passthrough.add_argument("--frame-height", type=int, required=True)
    passthrough.add_argument("--min-track-frames", type=int, default=5)
    passthrough.add_argument("--min-duration-sec", type=float, default=1.0)
    passthrough.add_argument("--border-margin-ratio", type=float, default=0.08)
    passthrough.add_argument("--min-displacement-norm", type=float, default=0.12)

    nlq = sub.add_parser("nlq", help="Simple natural-language query router")
    nlq.add_argument("--query", required=True)
    nlq.add_argument("--moments", default=None)
    nlq.add_argument("--tracks", default=None)
    nlq.add_argument("--frame-width", type=int, required=True)
    nlq.add_argument("--frame-height", type=int, required=True)
    nlq.add_argument(
        "--label-union-episodes",
        action="store_true",
        help="For 'appear' intent, merge all tracks of the label into shared presence windows",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir

    if args.command == "appear":
        moments = load_rows(_resolve(run_dir, args.moments, "moments.json"))
        tracks = load_rows(_resolve(run_dir, args.tracks, "normalized_tracks.json"))
        payload = {
            "label": args.label.strip().lower(),
            "appear_events": when_object_appears(moments, label=args.label),
            "episodes": appearance_episodes(
                tracks,
                label=args.label,
                max_gap_frames=int(args.max_gap_frames),
                min_episode_frames=int(args.min_episode_frames),
                per_track=not bool(args.label_union_episodes),
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "frames-with":
        tracks = load_rows(_resolve(run_dir, args.tracks, "normalized_tracks.json"))
        payload = frames_with_label(
            tracks,
            label=args.label,
            max_gap_frames=int(args.max_gap_frames),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "pass-through":
        tracks = load_rows(_resolve(run_dir, args.tracks, "normalized_tracks.json"))
        payload = pass_through_tracks(
            tracks,
            label=args.label,
            frame_width=int(args.frame_width),
            frame_height=int(args.frame_height),
            config=PassThroughConfig(
                min_track_frames=int(args.min_track_frames),
                min_duration_sec=float(args.min_duration_sec),
                border_margin_ratio=float(args.border_margin_ratio),
                min_displacement_norm=float(args.min_displacement_norm),
            ),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if args.command == "nlq":
        moments = load_rows(_resolve(run_dir, args.moments, "moments.json"))
        tracks = load_rows(_resolve(run_dir, args.tracks, "normalized_tracks.json"))
        payload = answer_nlq(
            args.query,
            moments=moments,
            tracks=tracks,
            frame_width=int(args.frame_width),
            frame_height=int(args.frame_height),
            per_track_episodes=not bool(args.label_union_episodes),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
