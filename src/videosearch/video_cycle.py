from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .moments import Moment, MomentConfig, generate_moments


TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass(slots=True)
class VideoManifest:
    video_id: str
    video_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    target_fps: float
    sampled_frame_count: int
    sampled_frames_path: str


@dataclass(slots=True)
class KeyframeRecord:
    moment_index: int
    role: str
    frame_idx: int
    time_sec: float
    image_path: str


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for video ingest/keyframe extraction. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


def ingest_video(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 10.0,
) -> VideoManifest:
    """
    Ingest a video and write sampled frames/metadata artifacts.
    """
    cv2 = _ensure_cv2()
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (frame_count / src_fps) if frame_count > 0 else 0.0

    sample_step = max(1, int(round(src_fps / target_fps)))
    sampled_rows: list[dict[str, Any]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_step == 0:
            t = idx / src_fps
            image_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(image_path), frame)
            sampled_rows.append(
                {
                    "frame_idx": idx,
                    "time_sec": round(t, 6),
                    "image_path": str(image_path),
                }
            )
        idx += 1
    cap.release()

    sampled_path = out_dir / "sampled_frames.json"
    sampled_path.write_text(json.dumps(sampled_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = VideoManifest(
        video_id=path.stem,
        video_path=str(path),
        width=width,
        height=height,
        fps=src_fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        target_fps=target_fps,
        sampled_frame_count=len(sampled_rows),
        sampled_frames_path=str(sampled_path),
    )
    (out_dir / "video_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        observations = raw.get("observations")
        if isinstance(observations, list):
            return [dict(item) for item in observations if isinstance(item, dict)]
    raise ValueError("Expected JSON list or object with `observations` list")


def video_fps(video_path: str | Path) -> float:
    cv2 = _ensure_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for FPS lookup: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps


def convert_bytetrack_mot_rows(
    mot_rows: Sequence[str],
    *,
    fps: float,
    class_label: str = "object",
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Convert ByteTrack MOT rows to the observation schema used by moment generation.

    Expected MOT row shape:
    frame,track_id,x,y,w,h,score,...
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    out: list[dict[str, Any]] = []
    clean_label = class_label.strip().lower() or "object"
    for raw in mot_rows:
        line = raw.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            frame_id_1 = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6]) if len(parts) >= 7 else 1.0
        except ValueError:
            continue

        if score < min_score:
            continue
        frame_idx = max(0, frame_id_1 - 1)  # MOT frame ids are 1-based.
        out.append(
            {
                "track_id": track_id,
                "class": clean_label,
                "bbox": [x, y, x + w, y + h],
                "confidence": score,
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps,
            }
        )

    out.sort(key=lambda row: (row["frame_idx"], row["time_sec"], row["track_id"]))
    return out


def convert_bytetrack_mot_file(
    mot_txt_path: str | Path,
    *,
    fps: float,
    class_label: str = "object",
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    rows = Path(mot_txt_path).read_text(encoding="utf-8").splitlines()
    return convert_bytetrack_mot_rows(
        rows,
        fps=fps,
        class_label=class_label,
        min_score=min_score,
    )


def load_synonym_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("synonym map must be a JSON object")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            continue
        out[str(key).strip().lower()] = value.strip().lower()
    return out


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def extract_object_nouns(
    captions: Sequence[str],
    min_count: int = 2,
    top_k: int = 64,
    stopwords: set[str] | None = None,
) -> list[str]:
    if min_count <= 0 or top_k <= 0:
        raise ValueError("min_count and top_k must be positive")
    stop = stopwords or DEFAULT_STOPWORDS
    counts: Counter[str] = Counter()
    for caption in captions:
        tokens = TOKEN_PATTERN.findall(caption.lower())
        for token in tokens:
            if token in stop or token.isdigit():
                continue
            clean = singularize(token)
            if len(clean) < 2:
                continue
            counts[clean] += 1

    nouns = [term for term, freq in counts.most_common(top_k) if freq >= min_count]
    return nouns


def load_captions(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("caption"), str):
                out.append(item["caption"])
        return out
    if isinstance(raw, dict):
        if isinstance(raw.get("captions"), list):
            out: list[str] = []
            for item in raw["captions"]:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and isinstance(item.get("caption"), str):
                    out.append(item["caption"])
            return out
    raise ValueError("Captions file must be a list of strings/objects or {captions: [...]}")


def canonicalize_label(label: str, synonym_map: Mapping[str, str]) -> str:
    token = singularize(label.strip().lower().replace("_", " "))
    return synonym_map.get(token, token)


def build_prompt_terms(
    seed_labels: Sequence[str],
    discovered_labels: Sequence[str],
    synonym_map: Mapping[str, str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in [*seed_labels, *discovered_labels]:
        clean = canonicalize_label(term, synonym_map)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def normalize_tracking_rows(
    rows: Iterable[Mapping[str, Any]],
    synonym_map: Mapping[str, str] | None = None,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Normalize DeepSORT/ByteTrack style rows into the moment module schema.
    """
    synonym_map = synonym_map or {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        track_id = row.get("track_id", row.get("id"))
        label = row.get("class", row.get("label", row.get("class_name", row.get("cls"))))
        bbox = row.get("bbox", row.get("xyxy"))
        confidence = row.get("confidence", row.get("score", row.get("conf", 0.0)))
        frame_idx = row.get("frame_idx", row.get("frame", row.get("frame_id")))
        time_sec = row.get("time_sec", row.get("timestamp", row.get("time")))

        if track_id is None or label is None or bbox is None or frame_idx is None:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        clean_label = canonicalize_label(str(label), synonym_map)
        if allowed_labels and clean_label not in allowed_labels:
            continue

        normalized.append(
            {
                "track_id": track_id,
                "class": clean_label,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": float(confidence),
                "frame_idx": int(frame_idx),
                "time_sec": float(time_sec) if time_sec is not None else 0.0,
            }
        )

    normalized.sort(key=lambda row: (row["frame_idx"], row["time_sec"], str(row["track_id"])))
    return normalized


def moments_to_dicts(moments: Sequence[Moment], video_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, moment in enumerate(moments):
        out.append(
            {
                "moment_index": idx,
                "video_id": video_id,
                "type": moment.type,
                "start_time": float(moment.start_time),
                "end_time": float(moment.end_time),
                "duration_sec": float(max(0.0, moment.end_time - moment.start_time)),
                "entities": list(moment.entities),
                "metadata": dict(moment.metadata),
            }
        )
    return out


def build_keyframe_targets(
    moments: Sequence[Moment] | Sequence[Mapping[str, Any]],
    fps: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")

    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(moments):
        if isinstance(raw, Moment):
            start = float(raw.start_time)
            end = float(raw.end_time)
        else:
            start = float(raw["start_time"])
            end = float(raw["end_time"])
        if end < start:
            end = start
        middle = (start + end) / 2.0
        slots = (("start", start), ("middle", middle), ("end", end))
        frames: list[dict[str, Any]] = []
        seen: set[int] = set()
        for role, t in slots:
            frame_idx = int(round(t * fps))
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            if frame_idx in seen:
                continue
            seen.add(frame_idx)
            frames.append({"role": role, "frame_idx": frame_idx, "time_sec": frame_idx / fps})
        out.append({"moment_index": idx, "frames": frames})
    return out


def extract_keyframes(
    video_path: str | Path,
    targets: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[KeyframeRecord]:
    cv2 = _ensure_cv2()
    video = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    records: list[KeyframeRecord] = []
    for target in targets:
        moment_index = int(target["moment_index"])
        frames = target.get("frames", [])
        if not isinstance(frames, list):
            continue
        for row in frames:
            frame_idx = int(row["frame_idx"])
            role = str(row["role"])
            time_sec = float(row["time_sec"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            path = out_dir / f"moment_{moment_index:05d}_{role}.jpg"
            cv2.imwrite(str(path), frame)
            records.append(
                KeyframeRecord(
                    moment_index=moment_index,
                    role=role,
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    image_path=str(path),
                )
            )
    cap.release()
    return records


def _histogram_image_embedding(image_path: str | Path, bins: int = 16) -> np.ndarray:
    cv2 = _ensure_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    hist_parts: list[np.ndarray] = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256]).flatten()
        hist_parts.append(hist.astype(np.float32))

    feature = np.concatenate(hist_parts, axis=0)
    mean = image.mean(axis=(0, 1)).astype(np.float32)
    std = image.std(axis=(0, 1)).astype(np.float32)
    feature = np.concatenate([feature, mean, std], axis=0)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature.astype(np.float32)


def build_moment_embeddings(
    keyframes: Sequence[KeyframeRecord],
    model_name: str = "histogram-rgb-16",
) -> tuple[dict[int, np.ndarray], str]:
    grouped: dict[int, list[np.ndarray]] = {}
    for record in keyframes:
        vector = _histogram_image_embedding(record.image_path)
        grouped.setdefault(record.moment_index, []).append(vector)

    pooled: dict[int, np.ndarray] = {}
    for moment_index, vectors in grouped.items():
        mat = np.vstack(vectors)
        mean = mat.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        pooled[moment_index] = mean.astype(np.float32)
    return pooled, model_name


def persist_moment_index(
    output_db: str | Path,
    moments: Sequence[Mapping[str, Any]],
    keyframes: Sequence[KeyframeRecord],
    embeddings: Mapping[int, np.ndarray],
    model_name: str,
) -> None:
    db_path = Path(output_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS moments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                moment_index INTEGER NOT NULL UNIQUE,
                video_id TEXT NOT NULL,
                type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                entities_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                keyframes_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS moment_vectors (
                moment_index INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                model_name TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_moments_type ON moments(type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_moments_video ON moments(video_id);")

        keyframe_by_moment: dict[int, list[dict[str, Any]]] = {}
        for row in keyframes:
            keyframe_by_moment.setdefault(row.moment_index, []).append(
                {
                    "role": row.role,
                    "frame_idx": row.frame_idx,
                    "time_sec": row.time_sec,
                    "image_path": row.image_path,
                }
            )

        for moment in moments:
            moment_index = int(moment["moment_index"])
            conn.execute(
                """
                INSERT OR REPLACE INTO moments(
                    moment_index, video_id, type, start_time, end_time,
                    entities_json, metadata_json, keyframes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment_index,
                    str(moment["video_id"]),
                    str(moment["type"]),
                    float(moment["start_time"]),
                    float(moment["end_time"]),
                    json.dumps(moment["entities"], ensure_ascii=True, sort_keys=True),
                    json.dumps(moment["metadata"], ensure_ascii=True, sort_keys=True),
                    json.dumps(keyframe_by_moment.get(moment_index, []), ensure_ascii=True, sort_keys=True),
                ),
            )

        for moment_index, vector in embeddings.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO moment_vectors(moment_index, embedding, embedding_dim, model_name)
                VALUES (?, ?, ?, ?)
                """,
                (int(moment_index), vector.tobytes(), int(vector.shape[0]), model_name),
            )
        conn.commit()


def _coerce_moment_config(overrides: Mapping[str, Any] | None) -> MomentConfig:
    if not overrides:
        return MomentConfig()
    base = asdict(MomentConfig())
    for key, value in overrides.items():
        if key in base:
            base[key] = value
    config = MomentConfig(**base)
    config.validate()
    return config


def _clamp_preview_limit(limit: int) -> int:
    return max(1, int(limit))


def _slice_for_report(
    rows: Sequence[Mapping[str, Any]] | Sequence[dict[str, Any]],
    *,
    include_full: bool,
    preview_limit: int,
) -> list[dict[str, Any]]:
    if include_full:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows[:preview_limit]]


def _moment_type_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("type", "UNKNOWN"))] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _embedding_preview(
    embeddings: Mapping[int, np.ndarray],
    *,
    include_full: bool,
    preview_limit: int,
) -> list[dict[str, Any]]:
    items = sorted(embeddings.items(), key=lambda item: item[0])
    if not include_full:
        items = items[:preview_limit]
    out: list[dict[str, Any]] = []
    for moment_index, vector in items:
        out.append(
            {
                "moment_index": int(moment_index),
                "l2_norm": float(np.linalg.norm(vector)),
                "vector_head": [float(value) for value in vector[:8]],
            }
        )
    return out


def build_phase_outputs(
    *,
    manifest: VideoManifest,
    sampled_rows: Sequence[Mapping[str, Any]],
    raw_tracks: Sequence[Mapping[str, Any]],
    normalized_tracks: Sequence[Mapping[str, Any]],
    moment_rows: Sequence[Mapping[str, Any]],
    keyframe_rows: Sequence[Mapping[str, Any]],
    embeddings: Mapping[int, np.ndarray],
    embedding_model_name: str,
    index_db_path: str | Path,
    synonym_map: Mapping[str, str],
    captions: Sequence[str] | None,
    discovered_labels: Sequence[str],
    prompt_terms: Sequence[str],
    include_full: bool = False,
    preview_limit: int = 25,
) -> dict[str, Any]:
    preview_limit = _clamp_preview_limit(preview_limit)
    captions = captions or []
    index_path = Path(index_db_path)
    db_counts = {"moments_table_rows": 0, "vectors_table_rows": 0}
    if index_path.exists():
        with sqlite3.connect(index_path) as conn:
            db_counts["moments_table_rows"] = int(conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0])
            db_counts["vectors_table_rows"] = int(conn.execute("SELECT COUNT(*) FROM moment_vectors").fetchone()[0])

    phase_outputs: dict[str, Any] = {
        "phase_1_ingest": {
            "manifest": asdict(manifest),
            "sampled_frames": _slice_for_report(
                sampled_rows,
                include_full=include_full,
                preview_limit=preview_limit,
            ),
            "sampled_frame_count": len(sampled_rows),
        },
        "phase_2_vocabulary": {
            "status": "applied" if captions else "skipped_no_captions",
            "captions_count": len(captions),
            "captions": list(captions if include_full else captions[:preview_limit]),
            "synonym_map": dict(synonym_map),
            "discovered_labels": list(discovered_labels),
            "prompt_terms": list(prompt_terms),
        },
        "phase_3_normalized_tracks": {
            "raw_track_row_count": len(raw_tracks),
            "normalized_track_row_count": len(normalized_tracks),
            "rows": _slice_for_report(
                normalized_tracks,
                include_full=include_full,
                preview_limit=preview_limit,
            ),
        },
        "phase_4_moments": {
            "moment_count": len(moment_rows),
            "moment_type_counts": _moment_type_counts(moment_rows),
            "moments": _slice_for_report(
                moment_rows,
                include_full=include_full,
                preview_limit=preview_limit,
            ),
        },
        "phase_5_keyframes": {
            "keyframe_count": len(keyframe_rows),
            "rows": _slice_for_report(
                keyframe_rows,
                include_full=include_full,
                preview_limit=preview_limit,
            ),
        },
        "phase_6_embeddings": {
            "embedding_model": embedding_model_name,
            "embedded_moment_count": len(embeddings),
            "embedding_dim": int(next(iter(embeddings.values())).shape[0]) if embeddings else 0,
            "rows": _embedding_preview(
                embeddings,
                include_full=include_full,
                preview_limit=preview_limit,
            ),
        },
        "phase_7_index": {
            "db_path": str(index_path),
            **db_counts,
        },
    }
    return phase_outputs


def run_video_cycle(
    video_path: str | Path,
    tracks_path: str | Path,
    output_dir: str | Path,
    *,
    captions_path: str | Path | None = None,
    synonyms_path: str | Path | None = None,
    seed_labels: Sequence[str] | None = None,
    target_fps: float = 10.0,
    moment_overrides: Mapping[str, Any] | None = None,
    include_full_phase_outputs: bool = False,
    phase_preview_limit: int = 25,
) -> dict[str, Any]:
    """
    End-to-end cycle:
    ingest video -> bootstrap vocabulary -> normalize tracks -> generate moments
    -> extract keyframes -> build embeddings -> persist SQLite index.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_preview_limit = _clamp_preview_limit(phase_preview_limit)

    manifest = ingest_video(video_path=video_path, output_dir=out_dir / "ingest", target_fps=target_fps)
    sampled_rows = json.loads(Path(manifest.sampled_frames_path).read_text(encoding="utf-8"))
    synonym_map = load_synonym_map(synonyms_path)

    prompts: list[str] = []
    discovered: list[str] = []
    captions: list[str] = []
    if captions_path:
        captions = load_captions(captions_path)
        discovered = extract_object_nouns(captions, min_count=1, top_k=128)
        prompts = build_prompt_terms(seed_labels or [], discovered, synonym_map)
        (out_dir / "vocabulary.json").write_text(
            json.dumps(
                {
                    "seed_labels": list(seed_labels or []),
                    "discovered_labels": discovered,
                    "prompt_terms": prompts,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    rows = load_json_rows(tracks_path)
    allowed_labels = set(prompts) if prompts else None
    normalized = normalize_tracking_rows(rows, synonym_map=synonym_map, allowed_labels=allowed_labels)

    normalized_path = out_dir / "normalized_tracks.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")

    moment_config = _coerce_moment_config(moment_overrides)
    moments = generate_moments(
        observations=normalized,
        frame_width=manifest.width,
        frame_height=manifest.height,
        config=moment_config,
    )
    moment_rows = moments_to_dicts(moments, video_id=manifest.video_id)
    moments_path = out_dir / "moments.json"
    moments_path.write_text(json.dumps(moment_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    targets = build_keyframe_targets(moment_rows, fps=manifest.fps, frame_count=max(manifest.frame_count, 1))
    keyframes = extract_keyframes(video_path=video_path, targets=targets, output_dir=out_dir / "moment_keyframes")
    keyframe_payload = [asdict(record) for record in keyframes]
    (out_dir / "moment_keyframes.json").write_text(
        json.dumps(keyframe_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    embeddings, model_name = build_moment_embeddings(keyframes)
    db_path = out_dir / "moment_index.sqlite"
    persist_moment_index(
        output_db=db_path,
        moments=moment_rows,
        keyframes=keyframes,
        embeddings=embeddings,
        model_name=model_name,
    )

    phase_outputs = build_phase_outputs(
        manifest=manifest,
        sampled_rows=sampled_rows,
        raw_tracks=rows,
        normalized_tracks=normalized,
        moment_rows=moment_rows,
        keyframe_rows=keyframe_payload,
        embeddings=embeddings,
        embedding_model_name=model_name,
        index_db_path=db_path,
        synonym_map=synonym_map,
        captions=captions,
        discovered_labels=discovered,
        prompt_terms=prompts,
        include_full=include_full_phase_outputs,
        preview_limit=phase_preview_limit,
    )
    phase_outputs_path = out_dir / "phase_outputs.json"
    phase_outputs_path.write_text(
        json.dumps(phase_outputs, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    summary = {
        "video_manifest": str(out_dir / "ingest" / "video_manifest.json"),
        "normalized_tracks": str(normalized_path),
        "moments": str(moments_path),
        "keyframes": str(out_dir / "moment_keyframes.json"),
        "moment_index_db": str(db_path),
        "phase_outputs": str(phase_outputs_path),
        "counts": {
            "tracks": len(normalized),
            "moments": len(moment_rows),
            "keyframes": len(keyframes),
            "embeddings": len(embeddings),
        },
    }
    if captions_path:
        summary["vocabulary"] = str(out_dir / "vocabulary.json")
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def cosine_search_moments(
    db_path: str | Path,
    query_vector: np.ndarray,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if query_vector.ndim != 1:
        raise ValueError("query_vector must be 1D")
    q = query_vector.astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                m.moment_index, m.video_id, m.type, m.start_time, m.end_time,
                m.entities_json, m.metadata_json, m.keyframes_json,
                v.embedding, v.embedding_dim, v.model_name
            FROM moments m
            JOIN moment_vectors v ON m.moment_index = v.moment_index
            """
        ).fetchall()

    if not rows:
        return []

    vectors = np.vstack([np.frombuffer(row[8], dtype=np.float32, count=int(row[9])) for row in rows])
    scores = vectors @ q
    order = np.argsort(-scores)[:top_k]
    results: list[dict[str, Any]] = []
    for idx in order:
        row = rows[int(idx)]
        results.append(
            {
                "score": float(scores[idx]),
                "moment_index": int(row[0]),
                "video_id": row[1],
                "type": row[2],
                "start_time": float(row[3]),
                "end_time": float(row[4]),
                "entities": json.loads(row[5]),
                "metadata": json.loads(row[6]),
                "keyframes": json.loads(row[7]),
                "model_name": row[10],
            }
        )
    return results
