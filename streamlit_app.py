from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import streamlit as st

from videosearch.moments import Moment, MomentConfig, TrackObservation, generate_moments
from videosearch.video_cycle import (
    DetectionTrackingConfig,
    LLMVocabPostprocessConfig,
    TrackProcessingConfig,
    VLMCaptionConfig,
    YOLOWorldConfig,
    run_video_cycle,
    semantic_search_moments,
)


SAMPLE_FILE = Path("examples/tracks/sample_static_scene.json")
DEFAULT_PHASE_OUTPUTS = Path("data/video_cycle_run/phase_outputs.json")
DEFAULT_RUN_DIR = Path("data/video_cycle_run")
DEFAULT_AUTO_LABELS = [
    "person",
    "car",
    "truck",
    "bus",
    "van",
    "motorcycle",
    "bicycle",
    "backpack",
    "suitcase",
]
DEFAULT_VLM_PROMPT = (
    "List visible objects and scene elements as a comma-separated list of singular nouns, "
    "lowercase, no adjectives/colors/verbs/locations, no duplicates, max 20 terms."
)


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_json_read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        if "phase_outputs" in payload and isinstance(payload["phase_outputs"], dict):
            return payload["phase_outputs"]
        return payload
    return None


def _safe_summary_read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(dict(item))
        return rows
    return []


def _render_image_grid(rows: list[dict[str, Any]], limit: int = 12) -> None:
    images: list[tuple[str, str]] = []
    for row in rows:
        image_path = row.get("image_path")
        if isinstance(image_path, str):
            path = Path(image_path)
            if path.exists():
                caption = f"frame={row.get('frame_idx', '?')} t={row.get('time_sec', '?')}"
                images.append((str(path), caption))
    if not images:
        st.info("No preview images found on disk for these rows.")
        return

    cols = st.columns(3)
    for idx, (img_path, caption) in enumerate(images[:limit]):
        with cols[idx % 3]:
            st.image(img_path, caption=caption, use_container_width=True)


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required to generate query clips. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


def _slug(text: str, *, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    if not cleaned:
        cleaned = "query"
    return cleaned[:max_len]


def _resolve_video_path_for_db(db_path: Path) -> Path | None:
    run_dir = db_path.parent
    summary_path = run_dir / "run_summary.json"
    summary = _safe_summary_read(summary_path)
    if isinstance(summary, Mapping):
        manifest_path = summary.get("video_manifest")
        if isinstance(manifest_path, str) and manifest_path.strip():
            manifest_file = _expand_path(manifest_path)
            payload = _safe_json_read(manifest_file)
            if isinstance(payload, Mapping):
                video_path = payload.get("video_path")
                if isinstance(video_path, str) and video_path.strip():
                    candidate = _expand_path(video_path)
                    if candidate.exists():
                        return candidate
    fallback_manifest = run_dir / "ingest" / "video_manifest.json"
    payload = _safe_json_read(fallback_manifest)
    if isinstance(payload, Mapping):
        video_path = payload.get("video_path")
        if isinstance(video_path, str) and video_path.strip():
            candidate = _expand_path(video_path)
            if candidate.exists():
                return candidate
    return None


def _export_query_clip(
    *,
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    padding_sec: float = 0.5,
) -> Path:
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
    duration = (frame_count / fps) if frame_count > 0 else 0.0

    start_sec = max(0.0, float(start_time) - float(padding_sec))
    end_sec = min(duration, float(end_time) + float(padding_sec)) if duration > 0 else float(end_time) + float(padding_sec)
    if end_sec <= start_sec:
        end_sec = start_sec + max(0.6, 2.0 / fps)

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame, int(end_sec * fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open clip writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    return output_path


def _build_query_clips(
    *,
    db_path: Path,
    query: str,
    top_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_dir = db_path.parent
    video_path = _resolve_video_path_for_db(db_path)
    if video_path is None:
        raise RuntimeError("Could not resolve source video path from run artifacts")

    query_dir = run_dir / "query_clips" / _slug(query)
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(top_results, start=1):
        moment_index = int(row.get("moment_index", -1))
        start_time = float(row.get("start_time", 0.0))
        end_time = float(row.get("end_time", start_time))
        clip_path = query_dir / f"rank_{rank:02d}_moment_{moment_index:05d}.mp4"
        _export_query_clip(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=clip_path,
            padding_sec=0.5,
        )
        out.append(
            {
                "rank": rank,
                "moment_index": moment_index,
                "start_time": start_time,
                "end_time": end_time,
                "clip_path": str(clip_path),
            }
        )
    return out


def _find_latest_index_db(base_dir: Path) -> Path | None:
    root = base_dir.expanduser().resolve()
    if not root.exists():
        return None
    candidates = list(root.glob("*/moment_index.sqlite"))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _list_existing_runs(base_dir: Path) -> list[Path]:
    root = base_dir.expanduser().resolve()
    if not root.exists():
        return []
    runs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "run_summary.json").exists() or (child / "phase_outputs.json").exists():
            runs.append(child)
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


def _resolve_runner_index_db() -> Path:
    summary = st.session_state.get("last_summary")
    if isinstance(summary, Mapping):
        db_text = summary.get("moment_index_db")
        if isinstance(db_text, str) and db_text.strip():
            db_path = _expand_path(db_text)
            if db_path.exists():
                return db_path

    run_dir_text = st.session_state.get("last_run_dir")
    if isinstance(run_dir_text, str) and run_dir_text.strip():
        db_path = _expand_path(run_dir_text) / "moment_index.sqlite"
        if db_path.exists():
            return db_path

    phase_outputs_text = st.session_state.get("last_phase_outputs_path")
    if isinstance(phase_outputs_text, str) and phase_outputs_text.strip():
        db_path = _expand_path(phase_outputs_text).parent / "moment_index.sqlite"
        if db_path.exists():
            return db_path

    latest = _find_latest_index_db(_expand_path("data/video_runs"))
    if latest is not None:
        return latest
    return _expand_path("data/video_runs/video_run/moment_index.sqlite")


def _render_semantic_query_panel(db_path: Path, *, key_prefix: str) -> None:
    st.markdown("### Semantic Query (Top 3)")

    query = st.text_input(
        "Ask a retrieval query",
        value="when does truck appear?",
        key=f"{key_prefix}_query_text",
    )
    search_clicked = st.button("Run Semantic Search", key=f"{key_prefix}_query_run")
    if search_clicked:
        if not db_path.exists():
            st.error(f"Index DB not found: `{db_path}`")
            return
        try:
            results = semantic_search_moments(db_path=str(db_path), query=query, top_k=3)
            st.session_state[f"{key_prefix}_query_results"] = results[:3]
            st.session_state[f"{key_prefix}_query_last"] = query
            st.session_state[f"{key_prefix}_query_clips"] = []
        except Exception as exc:
            st.error(f"Semantic search failed: {exc}")

    if not db_path.exists():
        st.caption(f"Set a valid index DB path or run pipeline first. Current path: `{db_path}`")
        return

    results = st.session_state.get(f"{key_prefix}_query_results", [])
    if not isinstance(results, list):
        results = []
    top_results = [row for row in results[:3] if isinstance(row, Mapping)]
    if not top_results:
        st.info("No query results yet.")
        return

    display_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    for row in top_results:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        display_rows.append(
            {
                "moment_index": int(row.get("moment_index", -1)),
                "score": round(float(row.get("score", 0.0)), 4),
                "type": str(row.get("type", "")),
                "start_time": round(float(row.get("start_time", 0.0)), 3),
                "end_time": round(float(row.get("end_time", 0.0)), 3),
                "label": str(metadata.get("label", metadata.get("label_group", ""))),
                "entities": ",".join(str(item) for item in row.get("entities", [])),
                "text_summary": str(row.get("text_summary", "")),
            }
        )
        keyframes = row.get("keyframes", [])
        if isinstance(keyframes, list):
            for item in keyframes:
                if isinstance(item, Mapping):
                    keyframe_rows.append(dict(item))

    st.dataframe(display_rows, use_container_width=True)
    st.download_button(
        "Download Top 3 Moments JSON",
        data=json.dumps(top_results, indent=2, ensure_ascii=True),
        file_name="semantic_top3_moments.json",
        mime="application/json",
        key=f"{key_prefix}_download_top3",
    )
    if keyframe_rows:
        st.write("Keyframe previews for top 3 results:")
        _render_image_grid(keyframe_rows, limit=9)

    st.markdown("#### Top 3 Moment Clips")
    current_query = st.session_state.get(f"{key_prefix}_query_last")
    if not isinstance(current_query, str) or not current_query.strip():
        current_query = query
    generate_clicked = st.button("Generate Top 3 Clips", key=f"{key_prefix}_generate_top3_clips")
    if generate_clicked:
        try:
            clip_rows = _build_query_clips(
                db_path=db_path,
                query=current_query,
                top_results=[dict(row) for row in top_results],
            )
            st.session_state[f"{key_prefix}_query_clips"] = clip_rows
        except Exception as exc:
            st.error(f"Clip generation failed: {exc}")

    clip_rows = st.session_state.get(f"{key_prefix}_query_clips", [])
    if isinstance(clip_rows, list) and clip_rows:
        for row in clip_rows:
            if not isinstance(row, Mapping):
                continue
            clip_path = row.get("clip_path")
            if not isinstance(clip_path, str):
                continue
            path = Path(clip_path)
            if not path.exists():
                continue
            try:
                start_t = float(row.get("start_time", 0.0))
                end_t = float(row.get("end_time", 0.0))
            except Exception:
                start_t = 0.0
                end_t = 0.0
            st.write(
                f"rank={row.get('rank')} moment={row.get('moment_index')} "
                f"time={start_t:.3f}s -> {end_t:.3f}s"
            )
            st.video(str(path))
            st.download_button(
                f"Download clip (rank {row.get('rank')})",
                data=path.read_bytes(),
                file_name=path.name,
                mime="video/mp4",
                key=f"{key_prefix}_clip_download_{row.get('rank')}_{row.get('moment_index')}",
            )


def _render_phase_payload(payload: Mapping[str, Any]) -> None:
    p1 = payload.get("phase_1_ingest", {})
    p2 = payload.get("phase_2_vocabulary", {})
    p3 = payload.get("phase_3_normalized_tracks", {})
    p4 = payload.get("phase_4_moments", {})
    p5 = payload.get("phase_5_keyframes", {})
    p6 = payload.get("phase_6_embeddings", {})
    p7 = payload.get("phase_7_index", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sampled Frames", int(p1.get("sampled_frame_count", 0)))
    c2.metric("Processed Tracks", int(p3.get("processed_track_row_count", p3.get("normalized_track_row_count", 0))))
    c3.metric("Moments", int(p4.get("moment_count", 0)))
    c4.metric("Embeddings", int(p6.get("embedded_moment_count", 0)))
    c5.metric("Semantic Embeds", int(p6.get("semantic_embedded_moment_count", 0)))

    tabs = st.tabs(
        [
            "Phase 1 Ingest",
            "Phase 2 Vocabulary",
            "Phase 3 Tracks",
            "Phase 4 Moments",
            "Phase 5 Keyframes",
            "Phase 6 Embeddings",
            "Phase 7 Index",
        ]
    )

    with tabs[0]:
        st.subheader("Manifest")
        st.json(p1.get("manifest", {}))
        sampled_rows = _as_rows(p1.get("sampled_frames"))
        st.subheader("Sampled Frames")
        if sampled_rows:
            st.dataframe(sampled_rows, use_container_width=True)
            _render_image_grid(sampled_rows, limit=9)
        else:
            st.info("No sampled frames listed.")

    with tabs[1]:
        st.subheader("Vocabulary")
        st.write(f"Status: `{p2.get('status', 'unknown')}`")
        st.write(f"Captions count: `{p2.get('captions_count', 0)}`")
        caption_rows = _as_rows(p2.get("caption_rows"))
        if caption_rows:
            st.write("Caption rows:")
            st.dataframe(caption_rows, use_container_width=True)
            st.write("VLM sampled frame previews:")
            _render_image_grid(caption_rows, limit=9)
        llm_post = p2.get("llm_postprocess", {})
        if isinstance(llm_post, dict) and llm_post:
            st.write("LLM postprocess:")
            st.json(llm_post)
        st.write("Discovered labels:")
        st.code(", ".join(p2.get("discovered_labels", [])) or "(none)")
        st.write("Prompt terms:")
        st.code(", ".join(p2.get("prompt_terms", [])) or "(none)")
        st.write("Synonym map:")
        st.json(p2.get("synonym_map", {}))

    with tabs[2]:
        st.subheader("Processed Tracks")
        st.write(f"Track source: `{p3.get('track_source', 'tracks_json')}`")
        st.write(f"Raw rows: `{p3.get('raw_track_row_count', 0)}`")
        st.write(f"Tracked rows: `{p3.get('tracked_row_count', p3.get('raw_track_row_count', 0))}`")
        st.write(f"Canonicalized rows: `{p3.get('canonicalized_track_row_count', p3.get('normalized_track_row_count', 0))}`")
        st.write(f"Processed rows: `{p3.get('processed_track_row_count', p3.get('normalized_track_row_count', 0))}`")
        detection_generation = p3.get("detection_generation", {})
        detection_tracking = p3.get("detection_tracking", {})
        track_processing = p3.get("track_processing", {})
        if isinstance(detection_generation, dict) and detection_generation:
            st.write("Detection generation report:")
            st.json(detection_generation)
        if isinstance(detection_tracking, dict) and detection_tracking:
            st.write("Detection tracking report:")
            st.json(detection_tracking)
        if isinstance(track_processing, dict) and track_processing:
            st.write("Track processing report:")
            st.json(track_processing)
        rows = _as_rows(p3.get("rows"))
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No track rows in report.")

    with tabs[3]:
        st.subheader("Moments")
        st.json(
            {
                "moment_type_counts": p4.get("moment_type_counts", {}),
                "color_tagged_moment_count": p4.get("color_tagged_moment_count", 0),
            }
        )
        rows = _as_rows(p4.get("moments"))
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No moments in report.")

    with tabs[4]:
        st.subheader("Moment Keyframes")
        rows = _as_rows(p5.get("rows"))
        if rows:
            st.dataframe(rows, use_container_width=True)
            _render_image_grid(rows, limit=12)
        else:
            st.info("No keyframes in report.")

    with tabs[5]:
        st.subheader("Embeddings")
        st.write(f"Image model: `{p6.get('embedding_model', 'unknown')}`")
        st.write(f"Image dim: `{p6.get('embedding_dim', 0)}`")
        st.write(f"Semantic model: `{p6.get('semantic_embedding_model', '') or '(disabled)'}`")
        st.write(f"Semantic dim: `{p6.get('semantic_embedding_dim', 0)}`")
        rows = _as_rows(p6.get("rows"))
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No embedding rows in report.")

    with tabs[6]:
        st.subheader("Index")
        st.json(p7)


def _render_pipeline_runner() -> None:
    st.title("Video Pipeline Runner")
    st.caption("Autopilot run: YOLO-World + tracking + VLM captions + moment index with minimal inputs.")

    existing_runs = _list_existing_runs(_expand_path("data/video_runs"))
    if existing_runs:
        run_options = [str(path) for path in existing_runs]
        selected_run = st.selectbox(
            "Existing runs",
            options=run_options,
            index=0,
            help="Load a previously processed run for phase inspection and semantic query.",
        )
        if st.button("Load Existing Run"):
            run_dir = _expand_path(selected_run)
            st.session_state["last_run_dir"] = str(run_dir)
            phase_path = run_dir / "phase_outputs.json"
            if phase_path.exists():
                st.session_state["last_phase_outputs_path"] = str(phase_path)
            summary_path = run_dir / "run_summary.json"
            summary = _safe_summary_read(summary_path)
            if summary is not None:
                st.session_state["last_summary"] = summary
            manifest_payload = _safe_json_read(run_dir / "ingest" / "video_manifest.json")
            if isinstance(manifest_payload, Mapping):
                video_path = manifest_payload.get("video_path")
                if isinstance(video_path, str) and video_path.strip():
                    st.session_state["last_video_name"] = Path(video_path).name
            st.success(f"Loaded existing run: {run_dir}")

    st.session_state.setdefault("last_video_name", "")
    uploaded_video = st.file_uploader(
        "Select video file",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        accept_multiple_files=False,
        help="Pick a local video file from your system.",
    )
    if uploaded_video is not None:
        st.session_state["last_video_name"] = str(uploaded_video.name)
    if st.session_state["last_video_name"]:
        st.caption(f"Selected: `{st.session_state['last_video_name']}`")

    c1, c2 = st.columns(2)
    with c1:
        vlm_endpoint = st.text_input("VLM endpoint", value="http://localhost:8000/v1/chat/completions")
        vlm_model = st.text_input("VLM model", value="nvidia/Qwen2.5-VL-7B-Instruct-NVFP4")
    with c2:
        yolo_device = st.text_input("Detector device", value="cuda")
        target_fps = st.number_input("Target FPS", min_value=1.0, max_value=120.0, value=10.0, step=1.0)
        vlm_frame_stride = st.number_input("VLM frame stride", min_value=1, max_value=500, value=10, step=1)

    with st.expander("Advanced (optional)", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            yolo_model = st.text_input("YOLO-World model", value="yolov8s-worldv2.pt")
            yolo_conf = st.number_input("YOLO confidence", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            yolo_iou = st.number_input("YOLO IOU threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            yolo_frame_stride = st.number_input("YOLO frame stride", min_value=1, max_value=200, value=1, step=1)
        with a2:
            vlm_prompt = st.text_area("VLM prompt", value=DEFAULT_VLM_PROMPT, height=96)
            semantic_embedder = st.selectbox("Semantic embedder", options=["hashing", "sentence-transformer"], index=0)
            semantic_model = st.text_input("Semantic model (optional)", value="")
            show_full_phase_outputs = st.checkbox("Include full phase outputs", value=False)
            phase_preview_limit = st.number_input("Phase preview limit", min_value=1, max_value=2000, value=25, step=1)
            log_progress = st.checkbox("Log progress", value=True)

    # Defaults used in autopilot mode.
    if "semantic_embedder" not in locals():
        semantic_embedder = "hashing"
    if "semantic_model" not in locals():
        semantic_model = ""
    if "show_full_phase_outputs" not in locals():
        show_full_phase_outputs = False
    if "phase_preview_limit" not in locals():
        phase_preview_limit = 25
    if "log_progress" not in locals():
        log_progress = True
    if "vlm_prompt" not in locals():
        vlm_prompt = DEFAULT_VLM_PROMPT
    if "yolo_model" not in locals():
        yolo_model = "yolov8s-worldv2.pt"
    if "yolo_conf" not in locals():
        yolo_conf = 0.3
    if "yolo_iou" not in locals():
        yolo_iou = 0.7
    if "yolo_frame_stride" not in locals():
        yolo_frame_stride = 1

    video_stem = "video_run"
    if uploaded_video is not None and str(uploaded_video.name).strip():
        video_stem = Path(str(uploaded_video.name)).stem or "video_run"
    elif st.session_state["last_video_name"]:
        video_stem = Path(str(st.session_state["last_video_name"])).stem or "video_run"
    out_dir_text = str((Path("data/video_runs") / video_stem).resolve())
    st.caption(f"Output directory (auto): `{out_dir_text}`")

    run_clicked = st.button("Run Full Pipeline", type="primary")
    if run_clicked:
        try:
            if uploaded_video is None:
                raise ValueError("Please select a video file before running.")

            out_dir = _expand_path(out_dir_text)
            out_dir.mkdir(parents=True, exist_ok=True)
            inputs_dir = out_dir / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(str(uploaded_video.name)).name or "video.mp4"
            video_path = inputs_dir / safe_name
            video_path.write_bytes(uploaded_video.getbuffer())

            yoloworld_cfg = YOLOWorldConfig(
                model=str(yolo_model),
                confidence=float(yolo_conf),
                iou_threshold=float(yolo_iou),
                device=str(yolo_device),
                frame_stride=int(yolo_frame_stride),
                max_frames=0,
            )
            yoloworld_cfg.validate()

            vlm_cfg = VLMCaptionConfig(
                endpoint=str(vlm_endpoint),
                model=str(vlm_model),
                prompt=str(vlm_prompt),
                max_tokens=120,
                frame_stride=int(vlm_frame_stride),
                timeout_sec=60.0,
                temperature=0.0,
                api_key=None,
            )
            vlm_cfg.validate()

            llm_vocab_cfg = LLMVocabPostprocessConfig(
                endpoint=str(vlm_endpoint),
                model=str(vlm_model),
                max_tokens=500,
                timeout_sec=60.0,
                temperature=0.0,
                api_key=None,
                max_detection_terms=20,
            )
            llm_vocab_cfg.validate()

            detection_tracking_cfg = DetectionTrackingConfig(
                iou_threshold=0.3,
                max_missed_frames=12,
                min_detection_confidence=0.2,
                class_aware=True,
            )
            detection_tracking_cfg.validate()

            track_processing_cfg = TrackProcessingConfig(
                min_confidence=0.0,
                min_track_length_frames=3,
                max_interp_gap_frames=1,
                clip_bboxes_to_frame=True,
            )
            track_processing_cfg.validate()

            with st.spinner("Running full video cycle. This can take a while on long videos..."):
                summary = run_video_cycle(
                    video_path=str(video_path),
                    tracks_path=None,
                    detections_path=None,
                    groundingdino_config=None,
                    yoloworld_config=yoloworld_cfg,
                    output_dir=str(out_dir),
                    detection_tracking_config=detection_tracking_cfg,
                    captions_path=None,
                    synonyms_path=None,
                    seed_labels=list(DEFAULT_AUTO_LABELS),
                    moment_label_allowlist=list(DEFAULT_AUTO_LABELS),
                    target_fps=float(target_fps),
                    moment_overrides=None,
                    track_processing_config=track_processing_cfg,
                    vlm_caption_config=vlm_cfg,
                    llm_vocab_postprocess_config=llm_vocab_cfg,
                    log_progress=bool(log_progress),
                    include_full_phase_outputs=bool(show_full_phase_outputs),
                    phase_preview_limit=int(phase_preview_limit),
                    semantic_index_embedder=semantic_embedder,
                    semantic_index_model=(semantic_model.strip() or None),
                    enable_semantic_index=True,
                )

            st.session_state["last_summary"] = summary
            st.session_state["last_phase_outputs_path"] = summary.get("phase_outputs")
            st.session_state["last_video_name"] = safe_name
            st.session_state["last_run_dir"] = str(out_dir)
            st.success("Pipeline run completed.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    summary = st.session_state.get("last_summary")
    if isinstance(summary, Mapping):
        st.markdown("### Run Summary")
        st.json(dict(summary))
        phase_path_text = summary.get("phase_outputs")
        if isinstance(phase_path_text, str):
            phase_path = _expand_path(phase_path_text)
            payload = _safe_json_read(phase_path)
            if payload is not None:
                st.markdown("### Phase Outputs")
                _render_phase_payload(payload)

    st.markdown("### Semantic Query")
    resolved_db_path = _resolve_runner_index_db()
    st.caption(f"Using index DB: `{resolved_db_path}`")
    _render_semantic_query_panel(resolved_db_path, key_prefix="runner")


def _render_cycle_inspector() -> None:
    st.title("Video Cycle Inspector")
    st.caption("Inspect existing `phase_outputs.json` generated by CLI or Pipeline Runner.")
    existing_runs = _list_existing_runs(_expand_path("data/video_runs"))
    if existing_runs:
        run_options = [str(path) for path in existing_runs]
        selected_run = st.selectbox(
            "Pick existing run",
            options=run_options,
            index=0,
            key="inspector_existing_run",
        )
        if st.button("Use Selected Run", key="inspector_use_selected_run"):
            selected_path = _expand_path(selected_run) / "phase_outputs.json"
            if selected_path.exists():
                st.session_state["last_phase_outputs_path"] = str(selected_path)
                st.session_state["last_run_dir"] = str(_expand_path(selected_run))
                summary = _safe_summary_read(_expand_path(selected_run) / "run_summary.json")
                if summary is not None:
                    st.session_state["last_summary"] = summary
                st.success(f"Loaded run: {selected_run}")

    default_path = st.session_state.get("last_phase_outputs_path") or str(DEFAULT_PHASE_OUTPUTS)
    phase_path_text = st.text_input("phase_outputs.json path", value=str(default_path))
    payload = _safe_json_read(_expand_path(phase_path_text))
    if payload is None:
        st.warning("Could not load phase outputs. Run pipeline first and point to a valid `phase_outputs.json`.")
        return

    summary_path = _expand_path(phase_path_text).parent / "run_summary.json"
    summary = _safe_summary_read(summary_path)
    if summary is not None:
        with st.expander("run_summary.json", expanded=False):
            st.json(summary)
    db_path = None
    if isinstance(summary, Mapping):
        db_text = summary.get("moment_index_db")
        if isinstance(db_text, str) and db_text.strip():
            db_path = _expand_path(db_text)
    if db_path is None:
        db_path = _expand_path(phase_path_text).parent / "moment_index.sqlite"
    _render_semantic_query_panel(db_path, key_prefix="inspector")
    _render_phase_payload(payload)


def _load_sample() -> dict[str, Any]:
    if not SAMPLE_FILE.exists():
        return {"frame_width": 1280, "frame_height": 720, "observations": []}
    payload = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"frame_width": 1280, "frame_height": 720, "observations": payload}
    if isinstance(payload, dict):
        return {
            "frame_width": int(payload.get("frame_width", 1280)),
            "frame_height": int(payload.get("frame_height", 720)),
            "observations": payload.get("observations", []),
        }
    return {"frame_width": 1280, "frame_height": 720, "observations": []}


def _parse_uploaded(raw_text: str) -> dict[str, Any]:
    data = json.loads(raw_text)
    if isinstance(data, list):
        return {"frame_width": 1280, "frame_height": 720, "observations": data}
    if isinstance(data, dict):
        observations = data.get("observations")
        if observations is None:
            raise ValueError("JSON object must contain an `observations` field")
        return {
            "frame_width": int(data.get("frame_width", 1280)),
            "frame_height": int(data.get("frame_height", 720)),
            "observations": observations,
        }
    raise ValueError("Uploaded JSON must be either a list or an object")


def _to_observations(rows: list[dict[str, Any]]) -> list[TrackObservation]:
    return [TrackObservation.from_mapping(row) for row in rows]


def _build_moment_config() -> MomentConfig:
    st.sidebar.header("Moment Config")
    appear = st.sidebar.number_input("APPEAR persist frames", min_value=1, max_value=30, value=5, step=1)
    disappear = st.sidebar.number_input("DISAPPEAR missing frames", min_value=1, max_value=60, value=10, step=1)
    stop_enter = st.sidebar.number_input("STOP enter frames", min_value=1, max_value=30, value=8, step=1)
    stop_exit = st.sidebar.number_input("STOP exit frames", min_value=1, max_value=30, value=8, step=1)
    near_enter = st.sidebar.number_input("NEAR enter frames", min_value=1, max_value=30, value=8, step=1)
    near_exit = st.sidebar.number_input("NEAR exit frames", min_value=1, max_value=30, value=8, step=1)
    approach_window = st.sidebar.number_input("APPROACH window", min_value=2, max_value=30, value=8, step=1)
    approach_reverse = st.sidebar.number_input("APPROACH reverse frames", min_value=1, max_value=30, value=8, step=1)
    stop_speed = st.sidebar.slider("STOP speed threshold", min_value=0.001, max_value=0.200, value=0.012, step=0.001)
    movement_speed = st.sidebar.slider("Movement speed threshold", min_value=0.001, max_value=0.300, value=0.022, step=0.001)
    near_threshold = st.sidebar.slider("NEAR threshold", min_value=0.010, max_value=0.500, value=0.090, step=0.001)
    near_exit_threshold = st.sidebar.slider("NEAR exit threshold", min_value=0.010, max_value=0.600, value=0.110, step=0.001)
    approach_drop = st.sidebar.slider("APPROACH drop threshold", min_value=0.005, max_value=0.500, value=0.040, step=0.001)
    ema_alpha = st.sidebar.slider("Speed EMA alpha", min_value=0.05, max_value=1.00, value=0.40, step=0.05)
    merge_gap = st.sidebar.slider("Merge gap (sec)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    config = MomentConfig(
        appear_persist_frames=int(appear),
        disappear_missing_frames=int(disappear),
        stop_enter_frames=int(stop_enter),
        stop_exit_frames=int(stop_exit),
        near_enter_frames=int(near_enter),
        near_exit_frames=int(near_exit),
        approach_window=int(approach_window),
        approach_reverse_frames=int(approach_reverse),
        stop_speed_threshold=float(stop_speed),
        movement_speed_threshold=float(movement_speed),
        near_threshold=float(near_threshold),
        near_threshold_exit=float(near_exit_threshold),
        approach_drop_threshold=float(approach_drop),
        speed_ema_alpha=float(ema_alpha),
        merge_gap_sec=float(merge_gap),
    )
    config.validate()
    return config


def _moment_rows(moments: list[Moment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for moment in moments:
        rows.append(
            {
                "type": moment.type,
                "start_time": round(moment.start_time, 3),
                "end_time": round(moment.end_time, 3),
                "duration_sec": round(max(0.0, moment.end_time - moment.start_time), 3),
                "entities": ",".join(str(entity) for entity in moment.entities),
                "metadata": json.dumps(moment.metadata, ensure_ascii=True, sort_keys=True),
            }
        )
    return rows


def _render_moment_lab() -> None:
    st.title("Moment Generator Lab")
    st.caption("Generate APPEAR, DISAPPEAR, STOP, NEAR, and APPROACH moments from tracking output.")

    source = st.radio("Input source", options=["Sample data", "Upload JSON"], horizontal=True)
    if source == "Sample data":
        payload = _load_sample()
    else:
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if not uploaded:
            st.info("Upload a JSON file to continue.")
            return
        try:
            payload = _parse_uploaded(uploaded.read().decode("utf-8"))
        except Exception as exc:
            st.error(f"Could not parse uploaded JSON: {exc}")
            return

    frame_width_default = int(payload.get("frame_width", 1280))
    frame_height_default = int(payload.get("frame_height", 720))
    observations_raw = payload.get("observations", [])

    c1, c2 = st.columns(2)
    with c1:
        frame_width = st.number_input("Frame width", min_value=1, max_value=10000, value=frame_width_default, step=1)
    with c2:
        frame_height = st.number_input("Frame height", min_value=1, max_value=10000, value=frame_height_default, step=1)

    st.write(f"Loaded rows: **{len(observations_raw)}**")
    if observations_raw:
        st.dataframe(observations_raw[: min(50, len(observations_raw))], use_container_width=True)

    try:
        config = _build_moment_config()
    except Exception as exc:
        st.error(f"Invalid config: {exc}")
        return

    if st.button("Generate Moments", type="primary"):
        try:
            moments = generate_moments(
                observations=_to_observations(observations_raw),
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                config=config,
            )
        except Exception as exc:
            st.error(f"Moment generation failed: {exc}")
            return

        st.success(f"Generated {len(moments)} moments")
        st.dataframe(_moment_rows(moments), use_container_width=True)
        st.download_button(
            "Download moments JSON",
            data=json.dumps(
                [
                    {
                        "type": row.type,
                        "start_time": row.start_time,
                        "end_time": row.end_time,
                        "entities": row.entities,
                        "metadata": row.metadata,
                    }
                    for row in moments
                ],
                indent=2,
                ensure_ascii=True,
            ),
            file_name="moments.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="LocalLens Dev UI", layout="wide")
    mode = st.sidebar.radio("View", options=["Pipeline Runner", "Cycle Inspector", "Moment Lab"], index=0)
    if mode == "Pipeline Runner":
        _render_pipeline_runner()
    elif mode == "Cycle Inspector":
        _render_cycle_inspector()
    else:
        _render_moment_lab()


if __name__ == "__main__":
    main()
