from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .moment_query import appearance_episodes, load_rows, when_object_appears


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for clip export. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


@dataclass(slots=True)
class ClipExportConfig:
    codec: str = "mp4v"
    padding_sec: float = 0.3
    overlay_boxes: bool = True
    line_thickness: int = 2
    font_scale: float = 0.5
    max_gap_frames: int = 2
    min_episode_frames: int = 2
    per_track_episodes: bool = True
    log_every_frames: int = 120

    def validate(self) -> None:
        if not self.codec.strip():
            raise ValueError("codec cannot be empty")
        if self.padding_sec < 0:
            raise ValueError("padding_sec must be >= 0")
        if self.line_thickness <= 0:
            raise ValueError("line_thickness must be > 0")
        if self.font_scale <= 0:
            raise ValueError("font_scale must be > 0")
        if self.max_gap_frames <= 0:
            raise ValueError("max_gap_frames must be > 0")
        if self.min_episode_frames <= 0:
            raise ValueError("min_episode_frames must be > 0")
        if self.log_every_frames <= 0:
            raise ValueError("log_every_frames must be > 0")


def build_label_episode_ranges(
    tracks: Sequence[Mapping[str, Any]],
    *,
    label: str,
    max_gap_frames: int,
    min_episode_frames: int,
    per_track: bool = True,
) -> list[dict[str, Any]]:
    return appearance_episodes(
        tracks,
        label=label,
        max_gap_frames=max_gap_frames,
        min_episode_frames=min_episode_frames,
        per_track=per_track,
    )


def _color_for_label(label: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) for ch in label) % 255
    return ((23 * seed + 31) % 255, (71 * seed + 17) % 255, (47 * seed + 97) % 255)


def _time_token(time_sec: float) -> str:
    return f"{time_sec:.3f}".replace(".", "p")


def _to_int(value: Any) -> Any:
    try:
        if isinstance(value, str) and value.strip():
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
    except Exception:
        return value
    return value


def _to_track_key(value: Any) -> str:
    coerced = _to_int(value)
    return str(coerced)


def resolve_color_track_ids(
    moments: Sequence[Mapping[str, Any]],
    *,
    label: str,
    color: str | None,
) -> set[str]:
    if color is None:
        return set()
    rows = when_object_appears(moments, label=label, color=color)
    out: set[str] = set()
    for row in rows:
        entities = row.get("entities", [])
        if not isinstance(entities, list):
            continue
        for entity in entities:
            out.add(_to_track_key(entity))
    return out


def export_label_episode_clips(
    *,
    video_path: str | Path,
    tracks_path: str | Path,
    moments_path: str | Path | None = None,
    label: str,
    color: str | None = None,
    output_dir: str | Path,
    config: ClipExportConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ClipExportConfig()
    cfg.validate()
    target = label.strip().lower()
    if not target:
        raise ValueError("label cannot be empty")

    tracks = load_rows(tracks_path)
    color_track_ids: set[str] = set()
    color_value = color.strip().lower() if isinstance(color, str) and color.strip() else None
    if color_value is not None:
        if moments_path is None:
            raise ValueError("moments_path is required when color filter is set")
        moments = load_rows(moments_path)
        color_track_ids = resolve_color_track_ids(moments, label=target, color=color_value)
        if color_track_ids:
            tracks = [
                row
                for row in tracks
                if str(row.get("class", "")).strip().lower() == target
                and _to_track_key(row.get("track_id")) in color_track_ids
            ]

    episodes = build_label_episode_ranges(
        tracks,
        label=target,
        max_gap_frames=cfg.max_gap_frames,
        min_episode_frames=cfg.min_episode_frames,
        per_track=cfg.per_track_episodes,
    )

    cv2 = _ensure_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 else 0.0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*cfg.codec)

    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in tracks:
        if str(row.get("class", "")).strip().lower() != target:
            continue
        frame_idx = int(row.get("frame_idx", 0))
        tracks_by_frame.setdefault(frame_idx, []).append(dict(row))

    clips: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        episode_start = float(episode["start_time"])
        episode_end = float(episode["end_time"])
        clip_start = max(0.0, episode_start - cfg.padding_sec)
        clip_end = min(duration_sec, episode_end + cfg.padding_sec)
        start_frame = max(0, int(clip_start * fps))
        end_frame = min(max(start_frame, int(clip_end * fps)), max(0, frame_count - 1))
        if end_frame < start_frame:
            continue

        token_start = _time_token(episode_start)
        token_end = _time_token(episode_end)
        out_path = out_dir / f"{target}_episode_{idx:03d}_{token_start}_{token_end}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {out_path}")

        episode_track_ids = {_to_int(value) for value in episode.get("track_ids", [])}
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        written = 0
        color = _color_for_label(target)
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.overlay_boxes:
                for row in tracks_by_frame.get(frame_idx, []):
                    track_id = _to_int(row.get("track_id"))
                    if episode_track_ids and track_id not in episode_track_ids:
                        continue
                    bbox = row.get("bbox", [0, 0, 0, 0])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.line_thickness)
                    caption = f"{target} id={track_id}"
                    cv2.putText(
                        frame,
                        caption,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale,
                        color,
                        max(1, cfg.line_thickness - 1),
                        cv2.LINE_AA,
                    )
            overlay = f"episode {idx:03d}  t={frame_idx / fps:.2f}s"
            cv2.putText(
                frame,
                overlay,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                overlay,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            written += 1
            if written == 1 or written % cfg.log_every_frames == 0:
                print(
                    f"[moment_clip] episode={idx} frame={frame_idx} written={written}/{(end_frame-start_frame+1)}",
                    flush=True,
                )
            frame_idx += 1

        writer.release()
        clips.append(
            {
                "episode_index": idx,
                "episode_start_time": episode_start,
                "episode_end_time": episode_end,
                "clip_start_time": clip_start,
                "clip_end_time": clip_end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "written_frames": written,
                "track_ids": list(episode_track_ids),
                "output_path": str(out_path),
            }
        )

    cap.release()
    summary = {
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "moments_path": (str(moments_path) if moments_path is not None else None),
        "label": target,
        "color": color_value,
        "color_track_ids": sorted(color_track_ids),
        "episode_count": len(episodes),
        "clip_count": len(clips),
        "fps": fps,
        "frame_size": {"width": width, "height": height},
        "config": asdict(cfg),
        "clips": clips,
    }
    summary_path = out_dir / f"{target}_episode_clips_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
