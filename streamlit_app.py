from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from videosearch.moments import Moment, MomentConfig, TrackObservation, generate_moments


SAMPLE_FILE = Path("examples/tracks/sample_static_scene.json")
DEFAULT_PHASE_OUTPUTS = Path("data/video_cycle_run/phase_outputs.json")


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
    parsed: list[TrackObservation] = []
    for row in rows:
        parsed.append(TrackObservation.from_mapping(row))
    return parsed


def _build_config() -> MomentConfig:
    st.sidebar.header("Moment Config")
    appear = st.sidebar.number_input("APPEAR persist frames", min_value=1, max_value=30, value=5, step=1)
    disappear = st.sidebar.number_input(
        "DISAPPEAR missing frames",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
    )
    stop_enter = st.sidebar.number_input("STOP enter frames", min_value=1, max_value=30, value=8, step=1)
    stop_exit = st.sidebar.number_input("STOP exit frames", min_value=1, max_value=30, value=8, step=1)
    near_enter = st.sidebar.number_input("NEAR enter frames", min_value=1, max_value=30, value=8, step=1)
    near_exit = st.sidebar.number_input("NEAR exit frames", min_value=1, max_value=30, value=8, step=1)
    approach_window = st.sidebar.number_input("APPROACH window", min_value=2, max_value=30, value=8, step=1)
    approach_reverse = st.sidebar.number_input(
        "APPROACH reverse frames",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
    )
    stop_speed = st.sidebar.slider("STOP speed threshold", min_value=0.001, max_value=0.200, value=0.012, step=0.001)
    movement_speed = st.sidebar.slider(
        "Movement speed threshold",
        min_value=0.001,
        max_value=0.300,
        value=0.022,
        step=0.001,
    )
    near_threshold = st.sidebar.slider("NEAR threshold", min_value=0.010, max_value=0.500, value=0.090, step=0.001)
    near_exit_threshold = st.sidebar.slider(
        "NEAR exit threshold",
        min_value=0.010,
        max_value=0.600,
        value=0.110,
        step=0.001,
    )
    approach_drop = st.sidebar.slider(
        "APPROACH drop threshold",
        min_value=0.005,
        max_value=0.500,
        value=0.040,
        step=0.001,
    )
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


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
        return out
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

    show = images[:limit]
    cols = st.columns(3)
    for idx, (img_path, caption) in enumerate(show):
        with cols[idx % 3]:
            st.image(img_path, caption=caption, use_container_width=True)


def _render_cycle_inspector() -> None:
    st.title("Video Cycle Inspector")
    st.caption("Inspect end-to-end pipeline phases from `phase_outputs.json`.")

    default_path = str(DEFAULT_PHASE_OUTPUTS)
    phase_path_str = st.text_input("phase_outputs.json path", value=default_path)
    phase_path = Path(phase_path_str)
    payload = _safe_json_read(phase_path)

    if payload is None:
        st.warning("Could not load phase outputs. Run the CLI first, then point this to `phase_outputs.json`.")
        st.code(
            "PYTHONPATH=src python -m videosearch.video_cycle_cli --video /path/video.mp4 "
            "--tracks /path/tracks.json --out-dir data/video_cycle_run --show-phase-outputs",
            language="bash",
        )
        return

    p1 = payload.get("phase_1_ingest", {})
    p2 = payload.get("phase_2_vocabulary", {})
    p3 = payload.get("phase_3_normalized_tracks", {})
    p4 = payload.get("phase_4_moments", {})
    p5 = payload.get("phase_5_keyframes", {})
    p6 = payload.get("phase_6_embeddings", {})
    p7 = payload.get("phase_7_index", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled Frames", p1.get("sampled_frame_count", 0))
    c2.metric("Normalized Tracks", p3.get("normalized_track_row_count", 0))
    c3.metric("Moments", p4.get("moment_count", 0))
    c4.metric("Embeddings", p6.get("embedded_moment_count", 0))

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
        st.subheader("Normalized Tracks")
        st.write(f"Raw rows: `{p3.get('raw_track_row_count', 0)}`")
        st.write(f"Canonicalized rows: `{p3.get('canonicalized_track_row_count', p3.get('normalized_track_row_count', 0))}`")
        st.write(f"Processed rows: `{p3.get('processed_track_row_count', p3.get('normalized_track_row_count', 0))}`")
        track_processing = p3.get("track_processing", {})
        if isinstance(track_processing, dict) and track_processing:
            st.write("Track processing report:")
            st.json(track_processing)
        track_rows = _as_rows(p3.get("rows"))
        if track_rows:
            st.dataframe(track_rows, use_container_width=True)
        else:
            st.info("No track rows in report.")

    with tabs[3]:
        st.subheader("Moments")
        st.json({"moment_type_counts": p4.get("moment_type_counts", {})})
        moment_rows = _as_rows(p4.get("moments"))
        if moment_rows:
            st.dataframe(moment_rows, use_container_width=True)
        else:
            st.info("No moments in report.")

    with tabs[4]:
        st.subheader("Moment Keyframes")
        keyframe_rows = _as_rows(p5.get("rows"))
        if keyframe_rows:
            st.dataframe(keyframe_rows, use_container_width=True)
            _render_image_grid(keyframe_rows, limit=12)
        else:
            st.info("No keyframes in report.")

    with tabs[5]:
        st.subheader("Embeddings")
        st.write(f"Model: `{p6.get('embedding_model', 'unknown')}`")
        st.write(f"Dimension: `{p6.get('embedding_dim', 0)}`")
        embed_rows = _as_rows(p6.get("rows"))
        if embed_rows:
            st.dataframe(embed_rows, use_container_width=True)
        else:
            st.info("No embedding rows in report.")

    with tabs[6]:
        st.subheader("Index")
        st.json(p7)


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
            raw_text = uploaded.read().decode("utf-8")
            payload = _parse_uploaded(raw_text)
        except Exception as exc:
            st.error(f"Could not parse uploaded JSON: {exc}")
            return

    frame_width_default = int(payload.get("frame_width", 1280))
    frame_height_default = int(payload.get("frame_height", 720))
    observations_raw = payload.get("observations", [])

    col1, col2 = st.columns(2)
    with col1:
        frame_width = st.number_input("Frame width", min_value=1, max_value=10000, value=frame_width_default, step=1)
    with col2:
        frame_height = st.number_input("Frame height", min_value=1, max_value=10000, value=frame_height_default, step=1)

    st.write(f"Loaded rows: **{len(observations_raw)}**")
    if observations_raw:
        st.dataframe(observations_raw[: min(50, len(observations_raw))], use_container_width=True)

    try:
        config = _build_config()
    except Exception as exc:
        st.error(f"Invalid config: {exc}")
        return

    if st.button("Generate Moments", type="primary"):
        try:
            observations = _to_observations(observations_raw)
            moments = generate_moments(
                observations=observations,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                config=config,
            )
        except Exception as exc:
            st.error(f"Moment generation failed: {exc}")
            return

        st.success(f"Generated {len(moments)} moments")
        rows = _moment_rows(moments)
        st.dataframe(rows, use_container_width=True)

        st.download_button(
            "Download moments JSON",
            data=json.dumps(
                [
                    {
                        "type": moment.type,
                        "start_time": moment.start_time,
                        "end_time": moment.end_time,
                        "entities": moment.entities,
                        "metadata": moment.metadata,
                    }
                    for moment in moments
                ],
                indent=2,
                ensure_ascii=True,
            ),
            file_name="moments.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="LocalLens Dev UI", layout="wide")
    mode = st.sidebar.radio("View", options=["Cycle Inspector", "Moment Lab"], index=0)
    if mode == "Cycle Inspector":
        _render_cycle_inspector()
    else:
        _render_moment_lab()


if __name__ == "__main__":
    main()
