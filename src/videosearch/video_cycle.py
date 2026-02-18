from __future__ import annotations

import base64
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as url_error
from urllib import request as url_request

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

DEFAULT_MOMENT_LABEL_ALLOWLIST = (
    "car",
    "truck",
    "bus",
    "van",
    "person",
    "motorcycle",
)


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


@dataclass(slots=True)
class VLMCaptionConfig:
    endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    prompt: str = "List all visible traffic objects in this traffic frame in one short sentence."
    max_tokens: int = 120
    frame_stride: int = 10
    timeout_sec: float = 60.0
    temperature: float = 0.0
    api_key: str | None = None

    def validate(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("VLM endpoint cannot be empty")
        if not self.model.strip():
            raise ValueError("VLM model cannot be empty")
        if not self.prompt.strip():
            raise ValueError("VLM prompt cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")


@dataclass(slots=True)
class LLMVocabPostprocessConfig:
    endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens: int = 500
    timeout_sec: float = 60.0
    temperature: float = 0.0
    api_key: str | None = None
    max_detection_terms: int = 20
    prompt_template: str | None = None

    def validate(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("LLM endpoint cannot be empty")
        if not self.model.strip():
            raise ValueError("LLM model cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        if self.max_detection_terms <= 0:
            raise ValueError("max_detection_terms must be > 0")


@dataclass(slots=True)
class TrackProcessingConfig:
    min_confidence: float = 0.0
    min_track_length_frames: int = 1
    max_interp_gap_frames: int = 0
    clip_bboxes_to_frame: bool = True

    def validate(self) -> None:
        if self.min_confidence < 0:
            raise ValueError("min_confidence must be >= 0")
        if self.min_track_length_frames <= 0:
            raise ValueError("min_track_length_frames must be > 0")
        if self.max_interp_gap_frames < 0:
            raise ValueError("max_interp_gap_frames must be >= 0")


@dataclass(slots=True)
class DetectionTrackingConfig:
    iou_threshold: float = 0.3
    max_missed_frames: int = 10
    min_detection_confidence: float = 0.0
    class_aware: bool = True

    def validate(self) -> None:
        if self.iou_threshold <= 0 or self.iou_threshold > 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0")
        if self.min_detection_confidence < 0:
            raise ValueError("min_detection_confidence must be >= 0")


@dataclass(slots=True)
class GroundingDINOConfig:
    model_config_path: str
    model_weights_path: str
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "cuda"
    frame_stride: int = 1
    max_frames: int = 0

    def validate(self) -> None:
        if not self.model_config_path.strip():
            raise ValueError("model_config_path cannot be empty")
        if not self.model_weights_path.strip():
            raise ValueError("model_weights_path cannot be empty")
        if self.box_threshold < 0 or self.box_threshold > 1:
            raise ValueError("box_threshold must be in [0, 1]")
        if self.text_threshold < 0 or self.text_threshold > 1:
            raise ValueError("text_threshold must be in [0, 1]")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        if self.max_frames < 0:
            raise ValueError("max_frames must be >= 0")


@dataclass(slots=True)
class YOLOWorldConfig:
    model: str = "yolov8s-worldv2.pt"
    confidence: float = 0.2
    iou_threshold: float = 0.7
    device: str = "cuda"
    frame_stride: int = 1
    max_frames: int = 0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.iou_threshold < 0 or self.iou_threshold > 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        if self.max_frames < 0:
            raise ValueError("max_frames must be >= 0")


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for video ingest/keyframe extraction. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


def _log_progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    print(f"[video_cycle] {message}", file=sys.stderr, flush=True)


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


def _image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image_bytes = path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def extract_chat_completion_text(response_payload: Mapping[str, Any]) -> str:
    """
    Parse OpenAI-compatible chat completion payload text.
    """
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"].strip())
        return " ".join(part for part in text_parts if part).strip()
    return ""


def _post_chat_completion(
    *,
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_sec: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = url_request.Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with url_request.urlopen(request, timeout=float(timeout_sec)) as response:
            raw = response.read().decode("utf-8")
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return json.loads(raw)


def _extract_json_object_from_text(text: str) -> dict[str, Any]:
    direct = text.strip()
    if direct:
        try:
            parsed = json.loads(direct)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON object from model output")


def _call_vlm_caption(config: VLMCaptionConfig, image_path: str | Path) -> tuple[str, dict[str, Any]]:
    image_data_url = _image_to_data_url(image_path)
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": int(config.max_tokens),
        "temperature": float(config.temperature),
    }
    parsed = _post_chat_completion(
        endpoint=config.endpoint,
        payload=payload,
        timeout_sec=config.timeout_sec,
        api_key=config.api_key,
    )
    text = extract_chat_completion_text(parsed)
    return text, parsed


def generate_vlm_captions(
    sampled_rows: Sequence[Mapping[str, Any]],
    config: VLMCaptionConfig,
    *,
    log_progress: bool = False,
) -> list[dict[str, Any]]:
    """
    Caption sampled frames through an OpenAI-compatible VLM endpoint.
    """
    config.validate()
    out: list[dict[str, Any]] = []
    stride = max(1, int(config.frame_stride))
    total_frames = len(sampled_rows)
    scheduled = (total_frames + stride - 1) // stride
    _log_progress(
        log_progress,
        f"Phase 2 VLM captions start: {scheduled} requests (sampled_frames={total_frames}, stride={stride})",
    )
    sent = 0

    for idx, row in enumerate(sampled_rows):
        if idx % stride != 0:
            continue
        sent += 1
        if sent == 1 or sent % 10 == 0 or sent == scheduled:
            _log_progress(log_progress, f"VLM progress: {sent}/{scheduled}")
        frame_idx = int(row.get("frame_idx", 0))
        time_sec = float(row.get("time_sec", 0.0))
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            continue

        entry: dict[str, Any] = {
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "image_path": image_path,
            "caption": "",
        }
        try:
            caption, _ = _call_vlm_caption(config, image_path)
            entry["caption"] = caption
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    _log_progress(log_progress, f"Phase 2 VLM captions done: {len(out)} rows")
    return out


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


def build_moment_label_allowlist(
    labels: Sequence[str] | None,
    synonym_map: Mapping[str, str],
) -> list[str]:
    source = labels if labels is not None else DEFAULT_MOMENT_LABEL_ALLOWLIST
    out: list[str] = []
    seen: set[str] = set()
    for label in source:
        clean = canonicalize_label(label, synonym_map)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def filter_labels_by_allowlist(
    labels: Sequence[str],
    allowlist: Sequence[str] | None,
) -> list[str]:
    if allowlist is None:
        return list(labels)
    allowed = set(allowlist)
    return [label for label in labels if label in allowed]


def _normalize_term_list(values: Any, *, max_items: int = 256) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _normalize_canonical_map(values: Any) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        src = key.strip().lower()
        dst = value.strip().lower()
        if not src or not dst:
            continue
        out[src] = dst
    return out


def build_vocab_postprocess_prompt(
    *,
    seed_labels: Sequence[str],
    discovered_labels: Sequence[str],
    prompt_terms: Sequence[str],
    max_detection_terms: int,
) -> str:
    seed = ", ".join(seed_labels) if seed_labels else "(none)"
    discovered = ", ".join(discovered_labels) if discovered_labels else "(none)"
    current = ", ".join(prompt_terms) if prompt_terms else "(none)"
    return (
        "Clean this vocabulary for open-vocab traffic detection.\n"
        f"Seed labels: {seed}\n"
        f"Discovered labels: {discovered}\n"
        f"Current prompt terms: {current}\n"
        "Rules: keep only concrete visual objects and infrastructure terms; "
        "remove adjectives, verbs, locations, weather words, and generic words.\n"
        "Return strict JSON only in this schema:\n"
        "{\"detection_terms\":[],\"scene_terms\":[],\"dropped_terms\":[],\"canonical_map\":{}}\n"
        f"Limit detection_terms to at most {int(max_detection_terms)}."
    )


def parse_vocab_postprocess_output(text: str, *, max_detection_terms: int = 20) -> dict[str, Any]:
    parsed = _extract_json_object_from_text(text)
    detection_terms = _normalize_term_list(parsed.get("detection_terms"), max_items=max_detection_terms)
    scene_terms = _normalize_term_list(parsed.get("scene_terms"))
    dropped_terms = _normalize_term_list(parsed.get("dropped_terms"))
    canonical_map = _normalize_canonical_map(parsed.get("canonical_map"))
    return {
        "detection_terms": detection_terms,
        "scene_terms": scene_terms,
        "dropped_terms": dropped_terms,
        "canonical_map": canonical_map,
    }


def postprocess_vocabulary_with_llm(
    *,
    seed_labels: Sequence[str],
    discovered_labels: Sequence[str],
    prompt_terms: Sequence[str],
    config: LLMVocabPostprocessConfig,
) -> dict[str, Any]:
    config.validate()
    prompt = config.prompt_template or build_vocab_postprocess_prompt(
        seed_labels=seed_labels,
        discovered_labels=discovered_labels,
        prompt_terms=prompt_terms,
        max_detection_terms=config.max_detection_terms,
    )

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(config.temperature),
        "max_tokens": int(config.max_tokens),
    }
    response = _post_chat_completion(
        endpoint=config.endpoint,
        payload=payload,
        timeout_sec=config.timeout_sec,
        api_key=config.api_key,
    )
    text = extract_chat_completion_text(response)
    normalized = parse_vocab_postprocess_output(
        text,
        max_detection_terms=config.max_detection_terms,
    )
    detection_terms = normalized["detection_terms"]
    scene_terms = normalized["scene_terms"]
    dropped_terms = normalized["dropped_terms"]
    canonical_map = normalized["canonical_map"]

    # Always preserve seed labels unless explicitly remapped.
    for seed in seed_labels:
        clean = seed.strip().lower()
        if not clean:
            continue
        canonical = canonical_map.get(clean, clean)
        if canonical not in detection_terms:
            detection_terms.append(canonical)
        if len(detection_terms) >= config.max_detection_terms:
            break

    detection_terms = detection_terms[: config.max_detection_terms]
    return {
        "status": "applied",
        "model": config.model,
        "endpoint": config.endpoint,
        "prompt": prompt,
        "detection_terms": detection_terms,
        "scene_terms": scene_terms,
        "dropped_terms": dropped_terms,
        "canonical_map": canonical_map,
        "raw_response_text": text,
    }


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def build_groundingdino_caption(prompt_terms: Sequence[str]) -> str:
    clean_terms = [term.strip().lower() for term in prompt_terms if term and term.strip()]
    if not clean_terms:
        raise ValueError("GroundingDINO prompt terms cannot be empty")
    # GroundingDINO commonly works best with phrase separators.
    return " . ".join(clean_terms) + " ."


def generate_groundingdino_detections(
    sampled_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_terms: Sequence[str],
    config: GroundingDINOConfig,
    log_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run GroundingDINO on sampled frames to produce frame-level detections.
    """
    config.validate()
    try:
        from groundingdino.util.inference import load_image, load_model, predict  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "groundingdino is required for --auto-detections-groundingdino. "
            "Install GroundingDINO in this environment first."
        ) from exc

    caption = build_groundingdino_caption(prompt_terms)
    model = load_model(config.model_config_path, config.model_weights_path)

    stride = max(1, int(config.frame_stride))
    scheduled = (len(sampled_rows) + stride - 1) // stride
    if config.max_frames > 0:
        scheduled = min(scheduled, config.max_frames)
    _log_progress(
        log_progress,
        f"Phase 3 detection start (GroundingDINO): frames={scheduled}, device={config.device}, "
        f"box_thr={config.box_threshold}, text_thr={config.text_threshold}",
    )
    detections: list[dict[str, Any]] = []
    frames_attempted = 0
    frames_processed = 0
    frames_failed = 0
    first_error: str | None = None

    for idx, row in enumerate(sampled_rows):
        if idx % stride != 0:
            continue
        if config.max_frames > 0 and frames_attempted >= config.max_frames:
            break
        frames_attempted += 1
        if frames_attempted == 1 or frames_attempted % 25 == 0 or frames_attempted == scheduled:
            _log_progress(log_progress, f"GroundingDINO progress: frame {frames_attempted}/{scheduled}")

        image_path = row.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            frames_failed += 1
            continue

        frame_idx = int(row.get("frame_idx", 0))
        time_sec = float(row.get("time_sec", 0.0))

        try:
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=caption,
                box_threshold=float(config.box_threshold),
                text_threshold=float(config.text_threshold),
                device=config.device,
            )
            h, w = image_source.shape[:2]
            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, bw, bh = [float(v) for v in box.tolist()]
                x1 = max(0.0, (cx - bw / 2.0) * w)
                y1 = max(0.0, (cy - bh / 2.0) * h)
                x2 = min(float(w), (cx + bw / 2.0) * w)
                y2 = min(float(h), (cy + bh / 2.0) * h)
                label = str(phrase).split("(", 1)[0].strip().lower()
                if not label:
                    continue
                detections.append(
                    {
                        "class": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(logit.item() if hasattr(logit, "item") else logit),
                        "frame_idx": frame_idx,
                        "time_sec": time_sec,
                    }
                )
            frames_processed += 1
        except Exception as exc:
            frames_failed += 1
            if first_error is None:
                first_error = str(exc)

    detections.sort(key=lambda item: (int(item["frame_idx"]), float(item["time_sec"]), str(item["class"])))
    class_counts = Counter(str(item["class"]) for item in detections)
    report = {
        "status": "generated",
        "frames_attempted": frames_attempted,
        "frames_processed": frames_processed,
        "frames_failed": frames_failed,
        "detection_row_count": len(detections),
        "class_row_counts": dict(sorted(class_counts.items(), key=lambda kv: kv[0])),
        "prompt_terms": list(prompt_terms),
        "caption": caption,
        "first_error": first_error,
        "config": asdict(config),
    }
    _log_progress(
        log_progress,
        f"Phase 3 detection done (GroundingDINO): detections={len(detections)}, "
        f"processed={frames_processed}, failed={frames_failed}",
    )
    return detections, report


def generate_yoloworld_detections(
    sampled_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_terms: Sequence[str],
    config: YOLOWorldConfig,
    log_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run YOLO-World on sampled frames to produce frame-level detections.
    """
    config.validate()
    try:
        from ultralytics import YOLOWorld  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for --auto-detections-yoloworld. "
            "Install with: pip install ultralytics"
        ) from exc

    clean_terms = [term.strip().lower() for term in prompt_terms if term and term.strip()]
    if not clean_terms:
        raise ValueError("YOLO-World prompt terms cannot be empty")

    model = YOLOWorld(config.model)
    model.set_classes(clean_terms)

    stride = max(1, int(config.frame_stride))
    scheduled = (len(sampled_rows) + stride - 1) // stride
    if config.max_frames > 0:
        scheduled = min(scheduled, config.max_frames)
    _log_progress(
        log_progress,
        f"Phase 3 detection start (YOLO-World): frames={scheduled}, device={config.device}, "
        f"model={config.model}, conf={config.confidence}",
    )
    detections: list[dict[str, Any]] = []
    frames_attempted = 0
    frames_processed = 0
    frames_failed = 0
    first_error: str | None = None

    for idx, row in enumerate(sampled_rows):
        if idx % stride != 0:
            continue
        if config.max_frames > 0 and frames_attempted >= config.max_frames:
            break
        frames_attempted += 1
        if frames_attempted == 1 or frames_attempted % 25 == 0 or frames_attempted == scheduled:
            _log_progress(log_progress, f"YOLO-World progress: frame {frames_attempted}/{scheduled}")

        image_path = row.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            frames_failed += 1
            if first_error is None:
                first_error = "missing_image_path"
            continue

        frame_idx = int(row.get("frame_idx", 0))
        time_sec = float(row.get("time_sec", 0.0))

        try:
            results = model.predict(
                source=image_path,
                conf=float(config.confidence),
                iou=float(config.iou_threshold),
                device=config.device,
                verbose=False,
            )
            if not results:
                frames_processed += 1
                continue
            result = results[0]
            names = result.names if hasattr(result, "names") else {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                frames_processed += 1
                continue
            xyxy = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.tolist() if hasattr(boxes, "conf") else []
            classes = boxes.cls.tolist() if hasattr(boxes, "cls") else []
            for i, bbox in enumerate(xyxy):
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                cls_idx = int(classes[i]) if i < len(classes) else -1
                raw_label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                label = str(raw_label).strip().lower()
                if not label:
                    continue
                confidence = float(confs[i]) if i < len(confs) else 0.0
                detections.append(
                    {
                        "class": label,
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "confidence": confidence,
                        "frame_idx": frame_idx,
                        "time_sec": time_sec,
                    }
                )
            frames_processed += 1
        except Exception as exc:
            frames_failed += 1
            if first_error is None:
                first_error = str(exc)

    detections.sort(key=lambda item: (int(item["frame_idx"]), float(item["time_sec"]), str(item["class"])))
    class_counts = Counter(str(item["class"]) for item in detections)
    report = {
        "status": "generated",
        "detector": "yolo_world",
        "frames_attempted": frames_attempted,
        "frames_processed": frames_processed,
        "frames_failed": frames_failed,
        "detection_row_count": len(detections),
        "class_row_counts": dict(sorted(class_counts.items(), key=lambda kv: kv[0])),
        "prompt_terms": clean_terms,
        "first_error": first_error,
        "config": asdict(config),
    }
    _log_progress(
        log_progress,
        f"Phase 3 detection done (YOLO-World): detections={len(detections)}, "
        f"processed={frames_processed}, failed={frames_failed}",
    )
    return detections, report


def normalize_detection_rows(
    rows: Iterable[Mapping[str, Any]],
    synonym_map: Mapping[str, str] | None = None,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Normalize frame-level detection rows into:
    class, bbox, confidence, frame_idx, time_sec.
    """
    synonym_map = synonym_map or {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("class", row.get("label", row.get("class_name", row.get("cls"))))
        bbox = row.get("bbox", row.get("xyxy"))
        confidence = row.get("confidence", row.get("score", row.get("conf", 0.0)))
        frame_idx = row.get("frame_idx", row.get("frame", row.get("frame_id")))
        time_sec = row.get("time_sec", row.get("timestamp", row.get("time")))

        if label is None or bbox is None or frame_idx is None:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        clean_label = canonicalize_label(str(label), synonym_map)
        if allowed_labels and clean_label not in allowed_labels:
            continue

        normalized.append(
            {
                "class": clean_label,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": float(confidence),
                "frame_idx": int(frame_idx),
                "time_sec": float(time_sec) if time_sec is not None else 0.0,
            }
        )

    normalized.sort(key=lambda row: (row["frame_idx"], row["time_sec"], row["class"]))
    return normalized


def track_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    config: DetectionTrackingConfig,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Lightweight class-aware IoU tracker to assign track IDs from detections.
    """
    config.validate()
    if fps <= 0:
        fps = 30.0

    input_rows = [dict(row) for row in detections]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    removed_low_confidence_count = 0
    for row in input_rows:
        conf = float(row.get("confidence", 0.0))
        if conf < config.min_detection_confidence:
            removed_low_confidence_count += 1
            continue
        frame_idx = int(row.get("frame_idx", 0))
        grouped[frame_idx].append(
            {
                "class": str(row.get("class", "object")),
                "bbox": [float(v) for v in row["bbox"]],
                "confidence": conf,
                "frame_idx": frame_idx,
                "time_sec": float(row.get("time_sec", frame_idx / fps)),
            }
        )

    active_tracks: dict[int, dict[str, Any]] = {}
    next_track_id = 1
    assigned_rows: list[dict[str, Any]] = []

    for frame_idx in sorted(grouped.keys()):
        frame_dets = grouped[frame_idx]

        stale_track_ids = [
            track_id
            for track_id, state in active_tracks.items()
            if (frame_idx - int(state["last_frame_idx"])) > config.max_missed_frames
        ]
        for track_id in stale_track_ids:
            del active_tracks[track_id]

        candidate_pairs: list[tuple[float, int, int]] = []
        track_ids = list(active_tracks.keys())
        for det_idx, det in enumerate(frame_dets):
            det_label = str(det["class"])
            for track_id in track_ids:
                state = active_tracks[track_id]
                if config.class_aware and str(state["class"]) != det_label:
                    continue
                iou = _bbox_iou_xyxy(state["bbox"], det["bbox"])
                if iou >= config.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matched: dict[int, int] = {}
        for _, track_id, det_idx in candidate_pairs:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            matched[det_idx] = track_id

        for det_idx, det in enumerate(frame_dets):
            track_id = matched.get(det_idx)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                active_tracks[track_id] = {
                    "class": det["class"],
                    "bbox": det["bbox"],
                    "last_frame_idx": frame_idx,
                    "hit_count": 1,
                }
            else:
                state = active_tracks[track_id]
                state["bbox"] = det["bbox"]
                state["last_frame_idx"] = frame_idx
                state["hit_count"] = int(state.get("hit_count", 0)) + 1

            assigned_rows.append(
                {
                    "track_id": int(track_id),
                    "class": str(det["class"]),
                    "bbox": [float(v) for v in det["bbox"]],
                    "confidence": float(det["confidence"]),
                    "frame_idx": int(frame_idx),
                    "time_sec": float(det.get("time_sec", frame_idx / fps)),
                }
            )

    assigned_rows.sort(key=lambda row: (row["frame_idx"], row["time_sec"], row["track_id"]))
    per_track_counts: Counter[int] = Counter(int(row["track_id"]) for row in assigned_rows)
    report = {
        "input_detection_row_count": len(input_rows),
        "removed_low_confidence_count": removed_low_confidence_count,
        "tracked_detection_row_count": len(assigned_rows),
        "track_count": len(per_track_counts),
        "track_length_stats": _track_length_stats(list(per_track_counts.values())),
        "config": asdict(config),
    }
    return assigned_rows, report


def _clamp_bbox_to_frame(bbox: Sequence[float], frame_width: int, frame_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    max_x = float(max(1, frame_width))
    max_y = float(max(1, frame_height))
    x1 = min(max(0.0, x1), max_x)
    y1 = min(max(0.0, y1), max_y)
    x2 = min(max(0.0, x2), max_x)
    y2 = min(max(0.0, y2), max_y)
    return [x1, y1, x2, y2]


def _track_length_stats(lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0}
    arr = np.array(lengths, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def process_tracking_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TrackProcessingConfig,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Phase 3 track processing:
    - deduplicate frame+track rows
    - confidence + bbox filtering
    - remove short tracks
    - optional temporal interpolation for short gaps
    """
    config.validate()
    if fps <= 0:
        fps = 30.0

    input_rows = [dict(item) for item in rows]
    input_count = len(input_rows)

    # Keep highest-confidence row when frame_idx and track_id collide.
    dedup: dict[tuple[int, Any], dict[str, Any]] = {}
    for row in input_rows:
        key = (int(row["frame_idx"]), row["track_id"])
        existing = dedup.get(key)
        if existing is None or float(row.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            dedup[key] = dict(row)
    dedup_rows = sorted(dedup.values(), key=lambda r: (int(r["frame_idx"]), float(r["time_sec"]), str(r["track_id"])))
    dedup_removed_count = input_count - len(dedup_rows)

    cleaned_rows: list[dict[str, Any]] = []
    removed_low_conf = 0
    removed_invalid_bbox = 0
    for row in dedup_rows:
        conf = float(row.get("confidence", 0.0))
        if conf < config.min_confidence:
            removed_low_conf += 1
            continue
        bbox = [float(v) for v in row["bbox"]]
        if config.clip_bboxes_to_frame:
            bbox = _clamp_bbox_to_frame(bbox, frame_width=frame_width, frame_height=frame_height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            removed_invalid_bbox += 1
            continue
        out = dict(row)
        out["confidence"] = conf
        out["bbox"] = bbox
        cleaned_rows.append(out)

    tracks_before: dict[Any, list[dict[str, Any]]] = {}
    for row in cleaned_rows:
        tracks_before.setdefault(row["track_id"], []).append(row)
    for obs in tracks_before.values():
        obs.sort(key=lambda r: int(r["frame_idx"]))

    short_track_ids: set[Any] = set()
    for track_id, obs in tracks_before.items():
        if len(obs) < config.min_track_length_frames:
            short_track_ids.add(track_id)

    removed_short_track_rows = sum(len(tracks_before[tid]) for tid in short_track_ids)
    tracks_kept: dict[Any, list[dict[str, Any]]] = {
        track_id: obs for track_id, obs in tracks_before.items() if track_id not in short_track_ids
    }

    interpolated_rows_added = 0
    if config.max_interp_gap_frames > 0:
        for track_id, obs in list(tracks_kept.items()):
            if len(obs) <= 1:
                continue
            expanded: list[dict[str, Any]] = [dict(obs[0])]
            for prev, curr in zip(obs, obs[1:]):
                prev_frame = int(prev["frame_idx"])
                curr_frame = int(curr["frame_idx"])
                gap = curr_frame - prev_frame
                if 1 < gap <= (config.max_interp_gap_frames + 1):
                    prev_bbox = [float(v) for v in prev["bbox"]]
                    curr_bbox = [float(v) for v in curr["bbox"]]
                    for step in range(1, gap):
                        alpha = step / gap
                        frame_idx = prev_frame + step
                        interp_bbox = [
                            prev_bbox[0] + alpha * (curr_bbox[0] - prev_bbox[0]),
                            prev_bbox[1] + alpha * (curr_bbox[1] - prev_bbox[1]),
                            prev_bbox[2] + alpha * (curr_bbox[2] - prev_bbox[2]),
                            prev_bbox[3] + alpha * (curr_bbox[3] - prev_bbox[3]),
                        ]
                        if config.clip_bboxes_to_frame:
                            interp_bbox = _clamp_bbox_to_frame(
                                interp_bbox,
                                frame_width=frame_width,
                                frame_height=frame_height,
                            )
                        interp = dict(prev)
                        interp["frame_idx"] = frame_idx
                        interp["time_sec"] = frame_idx / fps
                        interp["bbox"] = interp_bbox
                        interp["confidence"] = min(float(prev["confidence"]), float(curr["confidence"]))
                        interp["interpolated"] = True
                        expanded.append(interp)
                        interpolated_rows_added += 1
                expanded.append(dict(curr))
            expanded.sort(key=lambda r: int(r["frame_idx"]))
            tracks_kept[track_id] = expanded

    output_rows: list[dict[str, Any]] = []
    for obs in tracks_kept.values():
        output_rows.extend(obs)
    output_rows.sort(key=lambda r: (int(r["frame_idx"]), float(r["time_sec"]), str(r["track_id"])))

    track_lengths_before = [len(obs) for obs in tracks_before.values()]
    track_lengths_after = [len(obs) for obs in tracks_kept.values()]
    class_row_counts = Counter(str(row.get("class", "unknown")) for row in output_rows)
    class_track_counts = Counter(str(obs[0].get("class", "unknown")) for obs in tracks_kept.values() if obs)

    duration_sec = 0.0
    if output_rows:
        start_t = float(output_rows[0]["time_sec"])
        end_t = float(output_rows[-1]["time_sec"])
        duration_sec = max(1e-6, end_t - start_t)

    report = {
        "input_row_count": input_count,
        "dedup_removed_count": dedup_removed_count,
        "after_dedup_row_count": len(dedup_rows),
        "removed_low_confidence_count": removed_low_conf,
        "removed_invalid_bbox_count": removed_invalid_bbox,
        "removed_short_track_rows_count": removed_short_track_rows,
        "removed_short_track_count": len(short_track_ids),
        "interpolated_rows_added_count": interpolated_rows_added,
        "output_row_count": len(output_rows),
        "track_count_before_length_filter": len(tracks_before),
        "track_count_after_processing": len(tracks_kept),
        "track_length_stats_before": _track_length_stats(track_lengths_before),
        "track_length_stats_after": _track_length_stats(track_lengths_after),
        "class_row_counts": dict(sorted(class_row_counts.items(), key=lambda item: item[0])),
        "class_track_counts": dict(sorted(class_track_counts.items(), key=lambda item: item[0])),
        "rows_per_second": float(len(output_rows) / duration_sec) if duration_sec > 0 else 0.0,
        "config": asdict(config),
    }
    return output_rows, report


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
    *,
    log_progress: bool = False,
    progress_every: int = 100,
) -> list[KeyframeRecord]:
    cv2 = _ensure_cv2()
    video = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    total_requested = 0
    for target in targets:
        frames = target.get("frames", [])
        if isinstance(frames, list):
            total_requested += len(frames)

    _log_progress(log_progress, f"Phase 5 keyframe extraction start: requested_frames={total_requested}")

    records: list[KeyframeRecord] = []
    failed_reads = 0
    processed = 0
    log_step = max(1, int(progress_every))
    for target in targets:
        moment_index = int(target["moment_index"])
        frames = target.get("frames", [])
        if not isinstance(frames, list):
            continue
        for row in frames:
            processed += 1
            if processed == 1 or processed % log_step == 0 or processed == total_requested:
                _log_progress(log_progress, f"Phase 5 progress: {processed}/{total_requested} frame reads")
            frame_idx = int(row["frame_idx"])
            role = str(row["role"])
            time_sec = float(row["time_sec"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
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
    _log_progress(
        log_progress,
        f"Phase 5 keyframe extraction done: extracted={len(records)}, failed_reads={failed_reads}",
    )
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
    *,
    log_progress: bool = False,
    progress_every: int = 100,
) -> tuple[dict[int, np.ndarray], str]:
    _log_progress(log_progress, f"Phase 6 embedding start: keyframes={len(keyframes)}")
    grouped: dict[int, list[np.ndarray]] = {}
    log_step = max(1, int(progress_every))
    for idx, record in enumerate(keyframes, start=1):
        if idx == 1 or idx % log_step == 0 or idx == len(keyframes):
            _log_progress(log_progress, f"Phase 6 progress: {idx}/{len(keyframes)} keyframes embedded")
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
    _log_progress(log_progress, f"Phase 6 embedding done: moments_embedded={len(pooled)}")
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
    caption_rows: Sequence[Mapping[str, Any]] | None,
    discovered_labels: Sequence[str],
    prompt_terms: Sequence[str],
    moment_label_allowlist: Sequence[str] | None,
    phase2_status: str,
    track_source: str = "tracks_json",
    detection_generation_report: Mapping[str, Any] | None = None,
    tracked_rows: Sequence[Mapping[str, Any]] | None = None,
    detection_tracking_report: Mapping[str, Any] | None = None,
    canonicalized_tracks: Sequence[Mapping[str, Any]] | None = None,
    track_processing_report: Mapping[str, Any] | None = None,
    llm_vocab_postprocess: Mapping[str, Any] | None = None,
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
            "status": phase2_status,
            "captions_count": len(captions),
            "captions": list(captions if include_full else captions[:preview_limit]),
            "caption_rows": _slice_for_report(
                list(caption_rows or []),
                include_full=include_full,
                preview_limit=preview_limit,
            ),
            "synonym_map": dict(synonym_map),
            "discovered_labels": list(discovered_labels),
            "prompt_terms": list(prompt_terms),
            "moment_label_allowlist": list(moment_label_allowlist or []),
            "llm_postprocess": dict(llm_vocab_postprocess or {}),
        },
        "phase_3_normalized_tracks": {
            "track_source": track_source,
            "raw_track_row_count": len(raw_tracks),
            "tracked_row_count": len(tracked_rows or raw_tracks),
            "canonicalized_track_row_count": len(canonicalized_tracks or []),
            "processed_track_row_count": len(normalized_tracks),
            "normalized_track_row_count": len(normalized_tracks),
            "detection_generation": dict(detection_generation_report or {}),
            "detection_tracking": dict(detection_tracking_report or {}),
            "track_processing": dict(track_processing_report or {}),
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
    tracks_path: str | Path | None,
    output_dir: str | Path,
    *,
    detections_path: str | Path | None = None,
    groundingdino_config: GroundingDINOConfig | None = None,
    yoloworld_config: YOLOWorldConfig | None = None,
    detections_output_path: str | Path | None = None,
    detection_tracking_config: DetectionTrackingConfig | None = None,
    tracked_rows_output_path: str | Path | None = None,
    captions_path: str | Path | None = None,
    synonyms_path: str | Path | None = None,
    seed_labels: Sequence[str] | None = None,
    moment_label_allowlist: Sequence[str] | None = None,
    target_fps: float = 10.0,
    moment_overrides: Mapping[str, Any] | None = None,
    track_processing_config: TrackProcessingConfig | None = None,
    tracks_report_output_path: str | Path | None = None,
    vlm_caption_config: VLMCaptionConfig | None = None,
    captions_output_path: str | Path | None = None,
    llm_vocab_postprocess_config: LLMVocabPostprocessConfig | None = None,
    vocab_postprocess_output_path: str | Path | None = None,
    log_progress: bool = False,
    include_full_phase_outputs: bool = False,
    phase_preview_limit: int = 25,
) -> dict[str, Any]:
    """
    End-to-end cycle:
    ingest video -> bootstrap vocabulary -> (optional) detection tracking
    -> normalize/process tracks -> generate moments
    -> extract keyframes -> build embeddings -> persist SQLite index.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_preview_limit = _clamp_preview_limit(phase_preview_limit)
    _log_progress(log_progress, "Phase 1 ingest start")

    manifest = ingest_video(video_path=video_path, output_dir=out_dir / "ingest", target_fps=target_fps)
    sampled_rows = json.loads(Path(manifest.sampled_frames_path).read_text(encoding="utf-8"))
    synonym_map = load_synonym_map(synonyms_path)
    allowed_moment_labels = build_moment_label_allowlist(moment_label_allowlist, synonym_map)
    _log_progress(
        log_progress,
        f"Phase 1 ingest done: sampled_frames={len(sampled_rows)}, fps={manifest.fps:.2f}, duration={manifest.duration_sec:.2f}s",
    )

    prompts: list[str] = []
    discovered: list[str] = []
    caption_rows: list[dict[str, Any]] = []
    captions: list[str] = []
    phase2_status = "skipped_no_captions"
    llm_vocab_postprocess: dict[str, Any] = {"status": "skipped_not_requested"}
    effective_captions_path: str | Path | None = captions_path
    effective_vocab_postprocess_path: str | Path | None = None

    if captions_path:
        captions = load_captions(captions_path)
        phase2_status = "provided_captions"
    elif vlm_caption_config:
        caption_rows = generate_vlm_captions(
            sampled_rows=sampled_rows,
            config=vlm_caption_config,
            log_progress=log_progress,
        )
        effective_captions_path = captions_output_path or (out_dir / "vlm_captions_generated.json")
        effective_captions_file = Path(effective_captions_path)
        effective_captions_file.parent.mkdir(parents=True, exist_ok=True)
        effective_captions_file.write_text(
            json.dumps(
                {
                    "captions": caption_rows,
                    "vlm": {
                        "endpoint": vlm_caption_config.endpoint,
                        "model": vlm_caption_config.model,
                        "prompt": vlm_caption_config.prompt,
                        "max_tokens": vlm_caption_config.max_tokens,
                        "frame_stride": vlm_caption_config.frame_stride,
                        "timeout_sec": vlm_caption_config.timeout_sec,
                        "temperature": vlm_caption_config.temperature,
                    },
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        captions = [row["caption"] for row in caption_rows if isinstance(row.get("caption"), str) and row["caption"].strip()]
        phase2_status = "generated_vlm_captions"

    if llm_vocab_postprocess_config and not captions:
        llm_vocab_postprocess = {"status": "skipped_no_captions"}

    if captions:
        _log_progress(log_progress, f"Phase 2 vocabulary start: captions={len(captions)}")
        discovered = extract_object_nouns(captions, min_count=1, top_k=128)
        prompts = build_prompt_terms(seed_labels or [], discovered, synonym_map)

        if llm_vocab_postprocess_config:
            try:
                llm_vocab_postprocess = postprocess_vocabulary_with_llm(
                    seed_labels=seed_labels or [],
                    discovered_labels=discovered,
                    prompt_terms=prompts,
                    config=llm_vocab_postprocess_config,
                )
                post_terms = _normalize_term_list(
                    llm_vocab_postprocess.get("detection_terms"),
                    max_items=llm_vocab_postprocess_config.max_detection_terms,
                )
                if post_terms:
                    prompts = post_terms
                    phase2_status = f"{phase2_status}_with_llm_postprocess"
                else:
                    llm_vocab_postprocess["status"] = "failed_empty_detection_terms"
            except Exception as exc:
                llm_vocab_postprocess = {
                    "status": "failed",
                    "error": str(exc),
                }

            effective_vocab_postprocess_path = vocab_postprocess_output_path or (out_dir / "vocab_postprocess.json")
            post_file = Path(effective_vocab_postprocess_path)
            post_file.parent.mkdir(parents=True, exist_ok=True)
            post_file.write_text(
                json.dumps(llm_vocab_postprocess, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

        if allowed_moment_labels:
            prompts = filter_labels_by_allowlist(prompts, allowed_moment_labels)
            if not prompts:
                prompts = list(allowed_moment_labels)

        (out_dir / "vocabulary.json").write_text(
            json.dumps(
                {
                    "seed_labels": list(seed_labels or []),
                    "discovered_labels": discovered,
                    "prompt_terms": prompts,
                    "moment_label_allowlist": allowed_moment_labels,
                    "llm_postprocess": llm_vocab_postprocess,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        _log_progress(log_progress, f"Phase 2 vocabulary done: prompt_terms={len(prompts)}")

    if not prompts and allowed_moment_labels:
        prompts = list(allowed_moment_labels)
        _log_progress(
            log_progress,
            f"Phase 2 vocabulary skipped; using moment label allowlist as prompt_terms ({len(prompts)} labels)",
        )

    input_count = (
        int(bool(tracks_path))
        + int(bool(detections_path))
        + int(bool(groundingdino_config))
        + int(bool(yoloworld_config))
    )
    if input_count != 1:
        raise ValueError(
            "Provide exactly one of tracks_path, detections_path, groundingdino_config, or yoloworld_config"
        )

    rows: list[dict[str, Any]]
    track_source = "tracks_json"
    tracked_rows: list[dict[str, Any]] | None = None
    detection_tracking_report: dict[str, Any] | None = None
    detection_generation_report: dict[str, Any] | None = None

    allowed_labels = set(prompts) if prompts else None
    if groundingdino_config:
        track_source = "groundingdino_iou_tracker"
        generation_terms = prompts or list(seed_labels or [])
        if not generation_terms:
            generation_terms = ["car", "truck", "person"]
        generated_detections, detection_generation_report = generate_groundingdino_detections(
            sampled_rows=sampled_rows,
            prompt_terms=generation_terms,
            config=groundingdino_config,
            log_progress=log_progress,
        )
        effective_detections_path = detections_output_path or (out_dir / "groundingdino_detections_generated.json")
        detections_file = Path(effective_detections_path)
        detections_file.parent.mkdir(parents=True, exist_ok=True)
        detections_file.write_text(
            json.dumps(generated_detections, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        normalized_detections = normalize_detection_rows(
            generated_detections,
            synonym_map=synonym_map,
            allowed_labels=allowed_labels,
        )
        effective_tracking_config = detection_tracking_config or DetectionTrackingConfig()
        tracked_rows, detection_tracking_report = track_detections(
            normalized_detections,
            config=effective_tracking_config,
            fps=manifest.fps,
        )
        _log_progress(
            log_progress,
            f"Phase 3 tracking done: tracked_rows={len(tracked_rows)} (source={track_source})",
        )
        effective_tracked_path = tracked_rows_output_path or (out_dir / "tracked_rows.json")
        tracked_rows_file = Path(effective_tracked_path)
        tracked_rows_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_rows_file.write_text(
            json.dumps(tracked_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        rows = tracked_rows
    elif yoloworld_config:
        track_source = "yoloworld_iou_tracker"
        generation_terms = prompts or list(seed_labels or [])
        if not generation_terms:
            generation_terms = ["car", "truck", "person"]
        generated_detections, detection_generation_report = generate_yoloworld_detections(
            sampled_rows=sampled_rows,
            prompt_terms=generation_terms,
            config=yoloworld_config,
            log_progress=log_progress,
        )
        effective_detections_path = detections_output_path or (out_dir / "yoloworld_detections_generated.json")
        detections_file = Path(effective_detections_path)
        detections_file.parent.mkdir(parents=True, exist_ok=True)
        detections_file.write_text(
            json.dumps(generated_detections, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        normalized_detections = normalize_detection_rows(
            generated_detections,
            synonym_map=synonym_map,
            allowed_labels=allowed_labels,
        )
        effective_tracking_config = detection_tracking_config or DetectionTrackingConfig()
        tracked_rows, detection_tracking_report = track_detections(
            normalized_detections,
            config=effective_tracking_config,
            fps=manifest.fps,
        )
        _log_progress(
            log_progress,
            f"Phase 3 tracking done: tracked_rows={len(tracked_rows)} (source={track_source})",
        )
        effective_tracked_path = tracked_rows_output_path or (out_dir / "tracked_rows.json")
        tracked_rows_file = Path(effective_tracked_path)
        tracked_rows_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_rows_file.write_text(
            json.dumps(tracked_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        rows = tracked_rows
    elif detections_path:
        track_source = "detections_iou_tracker"
        detection_rows = load_json_rows(detections_path)
        normalized_detections = normalize_detection_rows(
            detection_rows,
            synonym_map=synonym_map,
            allowed_labels=allowed_labels,
        )
        effective_tracking_config = detection_tracking_config or DetectionTrackingConfig()
        tracked_rows, detection_tracking_report = track_detections(
            normalized_detections,
            config=effective_tracking_config,
            fps=manifest.fps,
        )
        _log_progress(
            log_progress,
            f"Phase 3 tracking done: tracked_rows={len(tracked_rows)} (source={track_source})",
        )
        effective_tracked_path = tracked_rows_output_path or (out_dir / "tracked_rows.json")
        tracked_rows_file = Path(effective_tracked_path)
        tracked_rows_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_rows_file.write_text(
            json.dumps(tracked_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        rows = tracked_rows
    else:
        rows = load_json_rows(tracks_path)
        tracked_rows = rows
        _log_progress(log_progress, f"Phase 3 input tracks loaded: rows={len(rows)}")

    canonicalized_tracks = normalize_tracking_rows(
        rows,
        synonym_map=synonym_map,
        allowed_labels=allowed_labels,
    )
    effective_track_config = track_processing_config or TrackProcessingConfig()
    normalized, track_processing_report = process_tracking_rows(
        canonicalized_tracks,
        config=effective_track_config,
        frame_width=manifest.width,
        frame_height=manifest.height,
        fps=manifest.fps,
    )
    _log_progress(log_progress, f"Phase 3 normalize/process done: rows={len(normalized)}")

    normalized_path = out_dir / "normalized_tracks.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
    effective_tracks_report_path = tracks_report_output_path or (out_dir / "tracks_report.json")
    tracks_report_file = Path(effective_tracks_report_path)
    tracks_report_file.parent.mkdir(parents=True, exist_ok=True)
    tracks_report_file.write_text(
        json.dumps(track_processing_report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    moment_config = _coerce_moment_config(moment_overrides)
    moments = generate_moments(
        observations=normalized,
        frame_width=manifest.width,
        frame_height=manifest.height,
        config=moment_config,
    )
    _log_progress(log_progress, f"Phase 4 moments done: count={len(moments)}")
    moment_rows = moments_to_dicts(moments, video_id=manifest.video_id)
    moments_path = out_dir / "moments.json"
    moments_path.write_text(json.dumps(moment_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    targets = build_keyframe_targets(moment_rows, fps=manifest.fps, frame_count=max(manifest.frame_count, 1))
    keyframes = extract_keyframes(
        video_path=video_path,
        targets=targets,
        output_dir=out_dir / "moment_keyframes",
        log_progress=log_progress,
    )
    keyframe_payload = [asdict(record) for record in keyframes]
    (out_dir / "moment_keyframes.json").write_text(
        json.dumps(keyframe_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    embeddings, model_name = build_moment_embeddings(keyframes, log_progress=log_progress)
    _log_progress(log_progress, f"Phase 5/6 keyframes+embeddings done: keyframes={len(keyframes)}, embeddings={len(embeddings)}")
    db_path = out_dir / "moment_index.sqlite"
    persist_moment_index(
        output_db=db_path,
        moments=moment_rows,
        keyframes=keyframes,
        embeddings=embeddings,
        model_name=model_name,
    )
    _log_progress(log_progress, "Phase 7 index done")

    phase_outputs = build_phase_outputs(
        manifest=manifest,
        sampled_rows=sampled_rows,
        raw_tracks=rows,
        tracked_rows=tracked_rows,
        track_source=track_source,
        detection_generation_report=detection_generation_report,
        detection_tracking_report=detection_tracking_report,
        normalized_tracks=normalized,
        canonicalized_tracks=canonicalized_tracks,
        track_processing_report=track_processing_report,
        moment_rows=moment_rows,
        keyframe_rows=keyframe_payload,
        embeddings=embeddings,
        embedding_model_name=model_name,
        index_db_path=db_path,
        synonym_map=synonym_map,
        captions=captions,
        caption_rows=caption_rows,
        discovered_labels=discovered,
        prompt_terms=prompts,
        moment_label_allowlist=allowed_moment_labels,
        phase2_status=phase2_status,
        llm_vocab_postprocess=llm_vocab_postprocess,
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
        "detections": (
            str(detections_output_path or (out_dir / "groundingdino_detections_generated.json"))
            if groundingdino_config
            else str(detections_output_path or (out_dir / "yoloworld_detections_generated.json"))
            if yoloworld_config
            else str(detections_path)
            if detections_path
            else None
        ),
        "normalized_tracks": str(normalized_path),
        "moments": str(moments_path),
        "keyframes": str(out_dir / "moment_keyframes.json"),
        "moment_index_db": str(db_path),
        "phase_outputs": str(phase_outputs_path),
        "tracked_rows": (
            str(tracked_rows_output_path or (out_dir / "tracked_rows.json"))
            if (detections_path or groundingdino_config or yoloworld_config)
            else None
        ),
        "captions": str(effective_captions_path) if effective_captions_path else None,
        "tracks_report": str(effective_tracks_report_path),
        "vocab_postprocess": str(effective_vocab_postprocess_path) if effective_vocab_postprocess_path else None,
        "counts": {
            "tracks": len(normalized),
            "moments": len(moment_rows),
            "keyframes": len(keyframes),
            "embeddings": len(embeddings),
        },
    }
    if captions:
        summary["vocabulary"] = str(out_dir / "vocabulary.json")
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _log_progress(log_progress, "Video cycle complete")
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
