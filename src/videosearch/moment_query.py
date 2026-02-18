from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        observations = raw.get("observations")
        if isinstance(observations, list):
            return [dict(item) for item in observations if isinstance(item, Mapping)]
    raise ValueError("Expected JSON list or object with `observations` list")


def _label_match(moment: Mapping[str, Any], label: str) -> bool:
    target = label.strip().lower()
    if not target:
        return False
    metadata = moment.get("metadata", {})
    if isinstance(metadata, Mapping):
        moment_label = str(metadata.get("label", "")).strip().lower()
        group = str(metadata.get("label_group", "")).strip().lower()
        if target in {moment_label, group}:
            return True
        pair = metadata.get("class_pair")
        if isinstance(pair, list):
            pair_set = {str(item).strip().lower() for item in pair}
            if target in pair_set:
                return True
    return False


def when_object_appears(
    moments: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in moments:
        if str(row.get("type", "")).upper() != "APPEAR":
            continue
        if not _label_match(row, label):
            continue
        out.append(
            {
                "moment_index": int(row.get("moment_index", -1)),
                "time_sec": float(row.get("start_time", 0.0)),
                "entities": list(row.get("entities", [])),
                "metadata": dict(row.get("metadata", {})),
            }
        )
    out.sort(key=lambda item: (item["time_sec"], item["moment_index"]))
    return out


def frames_with_label(
    tracks: Sequence[Mapping[str, Any]],
    *,
    label: str,
    max_gap_frames: int = 2,
) -> dict[str, Any]:
    target = label.strip().lower()
    if not target:
        return {"frames": [], "intervals": []}

    points: list[tuple[int, float]] = []
    for row in tracks:
        cls = str(row.get("class", "")).strip().lower()
        if cls != target:
            continue
        frame_idx = int(row.get("frame_idx", 0))
        time_sec = float(row.get("time_sec", 0.0))
        points.append((frame_idx, time_sec))
    points = sorted(set(points), key=lambda item: item[0])

    intervals: list[dict[str, Any]] = []
    if points:
        start_frame, start_time = points[0]
        prev_frame, prev_time = points[0]
        frame_count = 1
        for frame_idx, time_sec in points[1:]:
            if frame_idx <= prev_frame + max(1, int(max_gap_frames)):
                prev_frame, prev_time = frame_idx, time_sec
                frame_count += 1
                continue
            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": prev_frame,
                    "start_time": start_time,
                    "end_time": prev_time,
                    "frame_count": frame_count,
                }
            )
            start_frame, start_time = frame_idx, time_sec
            prev_frame, prev_time = frame_idx, time_sec
            frame_count = 1
        intervals.append(
            {
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "start_time": start_time,
                "end_time": prev_time,
                "frame_count": frame_count,
            }
        )

    frame_rows = [{"frame_idx": frame_idx, "time_sec": time_sec} for frame_idx, time_sec in points]
    return {"frames": frame_rows, "intervals": intervals}


def appearance_episodes(
    tracks: Sequence[Mapping[str, Any]],
    *,
    label: str,
    max_gap_frames: int = 2,
    min_episode_frames: int = 1,
) -> list[dict[str, Any]]:
    base = frames_with_label(tracks, label=label, max_gap_frames=max_gap_frames)
    min_frames = max(1, int(min_episode_frames))
    out: list[dict[str, Any]] = []
    target = label.strip().lower()
    for interval in base["intervals"]:
        frame_count = int(interval.get("frame_count", 0))
        if frame_count < min_frames:
            continue
        start_frame = int(interval["start_frame"])
        end_frame = int(interval["end_frame"])
        track_ids: set[Any] = set()
        for row in tracks:
            if str(row.get("class", "")).strip().lower() != target:
                continue
            frame_idx = int(row.get("frame_idx", 0))
            if start_frame <= frame_idx <= end_frame:
                track_ids.add(_to_simple_track_id(row.get("track_id")))
        out.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": float(interval["start_time"]),
                "end_time": float(interval["end_time"]),
                "duration_sec": max(0.0, float(interval["end_time"]) - float(interval["start_time"])),
                "frame_count": frame_count,
                "track_ids": sorted(track_ids, key=str),
            }
        )
    return out


def _to_simple_track_id(value: Any) -> Any:
    try:
        if isinstance(value, str) and value.strip():
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
    except Exception:
        return value
    return value


@dataclass(slots=True)
class PassThroughConfig:
    min_track_frames: int = 5
    min_duration_sec: float = 1.0
    border_margin_ratio: float = 0.08
    min_displacement_norm: float = 0.12

    def validate(self) -> None:
        if self.min_track_frames <= 0:
            raise ValueError("min_track_frames must be > 0")
        if self.min_duration_sec < 0:
            raise ValueError("min_duration_sec must be >= 0")
        if not (0.0 <= self.border_margin_ratio < 0.5):
            raise ValueError("border_margin_ratio must be in [0, 0.5)")
        if self.min_displacement_norm < 0:
            raise ValueError("min_displacement_norm must be >= 0")


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _near_border(point: tuple[float, float], width: float, height: float, margin_ratio: float) -> bool:
    x, y = point
    mx = margin_ratio * width
    my = margin_ratio * height
    return x <= mx or x >= (width - mx) or y <= my or y >= (height - my)


def pass_through_tracks(
    tracks: Sequence[Mapping[str, Any]],
    *,
    label: str,
    frame_width: int,
    frame_height: int,
    config: PassThroughConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or PassThroughConfig()
    cfg.validate()
    target = label.strip().lower()
    if not target:
        return []
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be > 0")

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        cls = str(row.get("class", "")).strip().lower()
        if cls != target:
            continue
        grouped[row.get("track_id")].append(dict(row))

    diag = (frame_width**2 + frame_height**2) ** 0.5
    out: list[dict[str, Any]] = []
    for track_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("frame_idx", 0)))
        if len(rows) < cfg.min_track_frames:
            continue
        start = rows[0]
        end = rows[-1]
        start_time = float(start.get("time_sec", 0.0))
        end_time = float(end.get("time_sec", 0.0))
        duration = max(0.0, end_time - start_time)
        if duration < cfg.min_duration_sec:
            continue

        c0 = _center(start.get("bbox", [0, 0, 0, 0]))
        c1 = _center(end.get("bbox", [0, 0, 0, 0]))
        disp = (((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5) / max(1e-6, diag)
        entered = _near_border(c0, frame_width, frame_height, cfg.border_margin_ratio)
        exited = _near_border(c1, frame_width, frame_height, cfg.border_margin_ratio)
        is_pass = bool((entered and exited) or (disp >= cfg.min_displacement_norm))

        if not is_pass:
            continue
        out.append(
            {
                "track_id": track_id,
                "label": target,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration,
                "start_frame": int(start.get("frame_idx", 0)),
                "end_frame": int(end.get("frame_idx", 0)),
                "entered_border": entered,
                "exited_border": exited,
                "displacement_norm": disp,
            }
        )
    out.sort(key=lambda item: (item["start_time"], item["track_id"]))
    return out


_KNOWN_LABELS = ("car", "truck", "bus", "van", "person", "motorcycle")


def _infer_label(query: str) -> str | None:
    q = query.lower()
    for label in _KNOWN_LABELS:
        if re.search(rf"\b{re.escape(label)}s?\b", q):
            return label
    return None


def answer_nlq(
    query: str,
    *,
    moments: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    q = query.strip().lower()
    label = _infer_label(q)
    if label is None:
        return {"intent": "unknown", "error": "Could not infer label from query"}

    if "appear" in q:
        appear_rows = when_object_appears(moments, label=label)
        episodes = appearance_episodes(tracks, label=label, max_gap_frames=2, min_episode_frames=2)
        return {
            "intent": "appear",
            "label": label,
            "results": appear_rows,
            "episodes": episodes,
        }
    if "pass" in q or "through" in q:
        return {
            "intent": "pass_through",
            "label": label,
            "results": pass_through_tracks(
                tracks,
                label=label,
                frame_width=frame_width,
                frame_height=frame_height,
            ),
        }
    if "frame" in q:
        return {
            "intent": "frames_with",
            "label": label,
            "results": frames_with_label(tracks, label=label),
        }
    return {"intent": "unknown", "error": "Could not infer query intent"}
