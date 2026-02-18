from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for overlay rendering. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    raise ValueError("Expected JSON list")


@dataclass(slots=True)
class OverlayConfig:
    codec: str = "mp4v"
    show_all_tracks: bool = False
    point_tolerance_frames: int = 1
    max_moment_lines: int = 6
    line_thickness: int = 2
    font_scale: float = 0.5
    start_sec: float = 0.0
    end_sec: float | None = None
    log_every_frames: int = 120

    def validate(self) -> None:
        if not self.codec.strip():
            raise ValueError("codec cannot be empty")
        if self.point_tolerance_frames < 0:
            raise ValueError("point_tolerance_frames must be >= 0")
        if self.max_moment_lines <= 0:
            raise ValueError("max_moment_lines must be > 0")
        if self.line_thickness <= 0:
            raise ValueError("line_thickness must be > 0")
        if self.font_scale <= 0:
            raise ValueError("font_scale must be > 0")
        if self.start_sec < 0:
            raise ValueError("start_sec must be >= 0")
        if self.end_sec is not None and self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        if self.log_every_frames <= 0:
            raise ValueError("log_every_frames must be > 0")


def _to_int_track_id(value: Any) -> Any:
    try:
        if isinstance(value, str) and value.strip():
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
    except Exception:
        return value
    return value


def active_moments_at_time(
    moments: Sequence[Mapping[str, Any]],
    *,
    time_sec: float,
    point_tolerance_sec: float = 0.0,
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for row in moments:
        start = float(row.get("start_time", 0.0))
        end = float(row.get("end_time", start))
        if end < start:
            end = start
        if abs(end - start) <= 1e-9:
            if abs(time_sec - start) <= point_tolerance_sec:
                active.append(dict(row))
        elif start <= time_sec <= end:
            active.append(dict(row))
    active.sort(key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("type", ""))))
    return active


def _class_color(label: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) for ch in label) % 255
    b = (37 * seed + 71) % 255
    g = (17 * seed + 149) % 255
    r = (97 * seed + 53) % 255
    return int(b), int(g), int(r)


def _moment_summary_lines(
    active_moments: Sequence[Mapping[str, Any]],
    *,
    max_lines: int,
) -> list[str]:
    if not active_moments:
        return ["moments: none"]
    counts: Counter[str] = Counter(str(row.get("type", "UNKNOWN")) for row in active_moments)
    lines = [f"moments active: {len(active_moments)}"]
    for moment_type, count in counts.most_common(max(1, max_lines - 1)):
        lines.append(f"- {moment_type}: {count}")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def render_moment_overlay_video(
    *,
    video_path: str | Path,
    tracks_path: str | Path,
    moments_path: str | Path,
    output_video_path: str | Path,
    config: OverlayConfig | None = None,
) -> dict[str, Any]:
    cfg = config or OverlayConfig()
    cfg.validate()
    cv2 = _ensure_cv2()

    tracks = load_json_rows(tracks_path)
    moments = load_json_rows(moments_path)

    frame_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        frame_idx = int(row.get("frame_idx", 0))
        frame_tracks[frame_idx].append(dict(row))
    for obs in frame_tracks.values():
        obs.sort(key=lambda item: str(item.get("track_id", "")))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / src_fps if frame_count > 0 else 0.0

    start_frame = int(max(0.0, cfg.start_sec) * src_fps)
    end_time = duration_sec if cfg.end_sec is None else min(duration_sec, float(cfg.end_sec))
    end_frame = max(start_frame, int(end_time * src_fps))

    fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0
    written = 0
    point_tol_sec = cfg.point_tolerance_frames / src_fps

    while True:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx > end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        time_sec = frame_idx / src_fps
        active = active_moments_at_time(moments, time_sec=time_sec, point_tolerance_sec=point_tol_sec)
        active_entities: set[Any] = set()
        for row in active:
            entities = row.get("entities", [])
            if isinstance(entities, list):
                for entity in entities:
                    active_entities.add(_to_int_track_id(entity))

        for obs in frame_tracks.get(frame_idx, []):
            track_id = _to_int_track_id(obs.get("track_id"))
            label = str(obs.get("class", "object"))
            if not cfg.show_all_tracks and track_id not in active_entities:
                continue
            bbox = obs.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            color = _class_color(label)
            if track_id in active_entities:
                color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.line_thickness)
            conf = float(obs.get("confidence", 0.0))
            caption = f"id={track_id} {label} {conf:.2f}"
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

        lines = [f"t={time_sec:.2f}s frame={frame_idx}"]
        lines.extend(_moment_summary_lines(active, max_lines=cfg.max_moment_lines))
        y = 24
        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            y += int(22 * cfg.font_scale + 8)

        writer.write(frame)
        written += 1
        processed += 1
        if processed == 1 or processed % cfg.log_every_frames == 0:
            print(f"[moment_overlay] processed={processed} written={written} frame={frame_idx}", flush=True)

    cap.release()
    writer.release()
    return {
        "output_video": str(out_path),
        "processed_frames": processed,
        "written_frames": written,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": src_fps,
        "width": width,
        "height": height,
    }
