from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from videosearch.moments import Moment, MomentConfig, TrackObservation, generate_moments


SAMPLE_FILE = Path("examples/tracks/sample_static_scene.json")


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


def main() -> None:
    st.set_page_config(page_title="Moment Generator Lab", layout="wide")
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


if __name__ == "__main__":
    main()

