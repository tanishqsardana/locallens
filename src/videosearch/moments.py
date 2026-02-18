from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
import math
import statistics
from typing import Any, Iterable, Mapping


TrackId = int | str
BBox = tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class TrackObservation:
    track_id: TrackId
    label: str
    bbox: BBox
    confidence: float
    frame_idx: int
    time_sec: float

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "TrackObservation":
        required = ("track_id", "class", "bbox", "confidence", "frame_idx", "time_sec")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing observation fields: {', '.join(missing)}")
        bbox_raw = data["bbox"]
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            raise ValueError("bbox must be a 4-value list or tuple: [x1, y1, x2, y2]")
        return TrackObservation(
            track_id=data["track_id"],
            label=str(data["class"]),
            bbox=(
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            ),
            confidence=float(data["confidence"]),
            frame_idx=int(data["frame_idx"]),
            time_sec=float(data["time_sec"]),
        )


@dataclass(slots=True)
class Moment:
    type: str
    start_time: float
    end_time: float
    entities: list[TrackId]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MomentConfig:
    appear_persist_frames: int = 5
    disappear_missing_frames: int = 10
    disappear_min_visible_frames: int = 5
    continuity_max_gap_frames: int = 3
    stop_enter_frames: int = 8
    stop_exit_frames: int = 8
    near_enter_frames: int = 8
    near_exit_frames: int = 8
    approach_window: int = 8
    approach_reverse_frames: int = 8
    stop_speed_threshold: float = 0.012
    movement_speed_threshold: float = 0.022
    near_threshold: float = 0.09
    near_threshold_exit: float = 0.11
    approach_drop_threshold: float = 0.04
    speed_ema_alpha: float = 0.4
    merge_gap_sec: float = 1.0
    emit_traffic_change: bool = True
    traffic_change_window_frames: int = 12
    traffic_change_threshold: int = 3
    traffic_change_cooldown_frames: int = 20
    traffic_change_labels: tuple[str, ...] = (
        "car",
        "truck",
        "bus",
        "van",
        "person",
        "motorcycle",
    )
    relevant_class_pairs: tuple[tuple[str, str], ...] = (
        ("person", "car"),
        ("person", "truck"),
        ("car", "truck"),
    )

    def validate(self) -> None:
        int_fields = (
            self.appear_persist_frames,
            self.disappear_missing_frames,
            self.disappear_min_visible_frames,
            self.continuity_max_gap_frames,
            self.stop_enter_frames,
            self.stop_exit_frames,
            self.near_enter_frames,
            self.near_exit_frames,
            self.approach_window,
            self.approach_reverse_frames,
            self.traffic_change_window_frames,
            self.traffic_change_threshold,
        )
        if any(value <= 0 for value in int_fields):
            raise ValueError("Frame-count thresholds must be > 0")
        if self.traffic_change_cooldown_frames < 0:
            raise ValueError("traffic_change_cooldown_frames must be >= 0")
        if self.near_threshold_exit <= self.near_threshold:
            raise ValueError("near_threshold_exit must be larger than near_threshold")
        if not (0.0 < self.speed_ema_alpha <= 1.0):
            raise ValueError("speed_ema_alpha must be in (0, 1]")
        if self.movement_speed_threshold <= self.stop_speed_threshold:
            raise ValueError("movement_speed_threshold must be > stop_speed_threshold")
        if self.merge_gap_sec < 0:
            raise ValueError("merge_gap_sec must be >= 0")

    @property
    def normalized_pairs(self) -> set[tuple[str, str]]:
        return {tuple(sorted(pair)) for pair in self.relevant_class_pairs}

    @property
    def traffic_change_label_set(self) -> set[str]:
        return {str(label).strip().lower() for label in self.traffic_change_labels if str(label).strip()}


@dataclass(slots=True)
class _TrackState:
    label: str
    label_group: str
    first_seen_frame: int
    first_seen_time: float
    last_seen_frame: int
    last_seen_time: float
    seen_run: int = 1
    total_seen_frames: int = 1
    appear_emitted: bool = False
    missing_count: int = 0
    prev_center: tuple[float, float] | None = None
    ema_speed: float | None = None
    below_count: int = 0
    below_start_time: float | None = None
    above_count: int = 0
    above_start_time: float | None = None
    active_stop_start: float | None = None
    min_stop_speed: float | None = None


@dataclass(slots=True)
class _PairState:
    entities: tuple[TrackId, TrackId]
    class_pair: tuple[str, str]
    near_active_start: float | None = None
    near_count: int = 0
    near_start_time: float | None = None
    far_count: int = 0
    far_start_time: float | None = None
    near_min_distance: float | None = None
    approach_active_start: float | None = None
    approach_reverse_count: int = 0
    approach_reverse_start_time: float | None = None
    approach_start_distance: float | None = None
    approach_min_distance: float | None = None
    distances: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    times: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    last_distance: float | None = None
    missing_count: int = 0


class MomentGenerator:
    """
    Build moments from detection+tracking outputs using temporal state transitions.
    This module is independent from model inference code.
    """

    def __init__(self, config: MomentConfig | None = None) -> None:
        self.config = config or MomentConfig()
        self.config.validate()

    def generate(
        self,
        observations: Iterable[TrackObservation | Mapping[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> list[Moment]:
        diag = math.hypot(frame_width, frame_height)
        if diag <= 0:
            raise ValueError("frame_width and frame_height must be positive")

        parsed = _parse_observations(observations)
        if not parsed:
            return []

        frame_map, frame_times = _build_frame_index(parsed)
        frame_indices = sorted(frame_map.keys())
        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]
        moments: list[Moment] = []

        tracks: dict[TrackId, _TrackState] = {}
        pair_states: dict[tuple[TrackId, TrackId], _PairState] = {}
        allowed_pairs = self.config.normalized_pairs
        traffic_labels = self.config.traffic_change_label_set
        traffic_history: deque[tuple[int, float, int]] = deque(maxlen=self.config.traffic_change_window_frames)
        last_traffic_change_frame = first_frame - self.config.traffic_change_cooldown_frames - 1

        for frame_idx in range(first_frame, last_frame + 1):
            current_time = frame_times[frame_idx]
            visible = frame_map.get(frame_idx, {})
            visible_ids = set(visible.keys())

            # Track-level updates for visible entities.
            for track_id, obs in visible.items():
                center = _bbox_center(obs.bbox)
                group = _class_group(obs.label)
                if track_id not in tracks:
                    tracks[track_id] = _TrackState(
                        label=obs.label,
                        label_group=group,
                        first_seen_frame=frame_idx,
                        first_seen_time=current_time,
                        last_seen_frame=frame_idx,
                        last_seen_time=current_time,
                        prev_center=center,
                    )
                    state = tracks[track_id]
                else:
                    state = tracks[track_id]
                    prev_frame = state.last_seen_frame
                    prev_time = state.last_seen_time
                    prev_center = state.prev_center
                    frame_gap = frame_idx - prev_frame

                    if 1 <= frame_gap <= self.config.continuity_max_gap_frames:
                        state.seen_run += 1
                    else:
                        if state.active_stop_start is not None:
                            _append_moment(
                                moments,
                                "STOP",
                                state.active_stop_start,
                                prev_time,
                                [track_id],
                                {
                                    "label": state.label,
                                    "label_group": state.label_group,
                                    "min_speed": state.min_stop_speed,
                                },
                            )
                            state.active_stop_start = None
                            state.min_stop_speed = None
                        state.seen_run = 1
                        state.first_seen_frame = frame_idx
                        state.first_seen_time = current_time
                        state.ema_speed = None
                        state.below_count = 0
                        state.below_start_time = None
                        state.above_count = 0
                        state.above_start_time = None

                    state.missing_count = 0
                    state.total_seen_frames += 1
                    state.label = obs.label
                    state.label_group = group

                    if (
                        1 <= frame_gap <= self.config.continuity_max_gap_frames
                        and prev_center is not None
                        and current_time > prev_time
                    ):
                        raw_speed = (
                            _center_distance(center, prev_center)
                            / diag
                            / (current_time - prev_time)
                        )
                        if state.ema_speed is None:
                            state.ema_speed = raw_speed
                        else:
                            alpha = self.config.speed_ema_alpha
                            state.ema_speed = alpha * raw_speed + (1.0 - alpha) * state.ema_speed
                    else:
                        state.ema_speed = None
                        state.below_count = 0
                        state.below_start_time = None
                        state.above_count = 0
                        state.above_start_time = None

                    self._update_stop_state(
                        moments=moments,
                        state=state,
                        current_time=current_time,
                        track_id=track_id,
                    )

                    state.last_seen_frame = frame_idx
                    state.last_seen_time = current_time
                    state.prev_center = center

                if not state.appear_emitted and state.seen_run >= self.config.appear_persist_frames:
                    _append_moment(
                        moments,
                        "APPEAR",
                        state.first_seen_time,
                        state.first_seen_time,
                        [track_id],
                        {
                            "label": state.label,
                            "label_group": state.label_group,
                            "persisted_frames": state.seen_run,
                        },
                    )
                    state.appear_emitted = True

            # Track-level missing updates.
            for track_id, state in tracks.items():
                if track_id in visible_ids:
                    continue
                state.missing_count += 1
                if state.active_stop_start is not None and state.missing_count == 1:
                    _append_moment(
                        moments,
                        "STOP",
                        state.active_stop_start,
                        state.last_seen_time,
                        [track_id],
                        {
                            "label": state.label,
                            "label_group": state.label_group,
                            "min_speed": state.min_stop_speed,
                        },
                    )
                    state.active_stop_start = None
                    state.min_stop_speed = None
                    state.below_count = 0
                    state.below_start_time = None
                    state.above_count = 0
                    state.above_start_time = None
                if state.missing_count == self.config.disappear_missing_frames:
                    if state.total_seen_frames >= self.config.disappear_min_visible_frames:
                        _append_moment(
                            moments,
                            "DISAPPEAR",
                            current_time,
                            current_time,
                            [track_id],
                            {
                                "label": state.label,
                                "label_group": state.label_group,
                            },
                        )

            if self.config.emit_traffic_change:
                dynamic_visible_count = sum(
                    1
                    for track_id in visible_ids
                    if tracks.get(track_id) is not None and tracks[track_id].label_group in traffic_labels
                )
                traffic_history.append((frame_idx, current_time, dynamic_visible_count))
                if len(traffic_history) == self.config.traffic_change_window_frames:
                    first_frame_idx, first_time, first_count = traffic_history[0]
                    _, last_time, last_count = traffic_history[-1]
                    delta = int(last_count - first_count)
                    if (
                        abs(delta) >= self.config.traffic_change_threshold
                        and (frame_idx - last_traffic_change_frame) >= self.config.traffic_change_cooldown_frames
                    ):
                        direction = "increase" if delta > 0 else "decrease"
                        _append_moment(
                            moments,
                            "TRAFFIC_CHANGE",
                            float(first_time),
                            float(last_time),
                            [],
                            {
                                "from_count": int(first_count),
                                "to_count": int(last_count),
                                "delta": delta,
                                "direction": direction,
                                "window_frames": self.config.traffic_change_window_frames,
                                "start_frame": int(first_frame_idx),
                                "end_frame": int(frame_idx),
                            },
                        )
                        last_traffic_change_frame = frame_idx

            # Pair-level updates.
            seen_pairs: set[tuple[TrackId, TrackId]] = set()
            for id_a, id_b in combinations(sorted(visible_ids, key=str), 2):
                state_a = tracks[id_a]
                state_b = tracks[id_b]
                pair_group = tuple(sorted((state_a.label_group, state_b.label_group)))
                if pair_group not in allowed_pairs:
                    continue
                pair_key = tuple(sorted((id_a, id_b), key=str))
                seen_pairs.add(pair_key)
                pair_state = pair_states.get(pair_key)
                if pair_state is None:
                    pair_state = _PairState(
                        entities=pair_key,
                        class_pair=pair_group,
                        distances=deque(maxlen=self.config.approach_window),
                        times=deque(maxlen=self.config.approach_window),
                    )
                    pair_states[pair_key] = pair_state

                pair_state.missing_count = 0
                center_a = _bbox_center(visible[id_a].bbox)
                center_b = _bbox_center(visible[id_b].bbox)
                distance = _center_distance(center_a, center_b) / diag
                pair_state.last_distance = distance
                pair_state.distances.append(distance)
                pair_state.times.append(current_time)

                self._update_near_state(
                    moments=moments,
                    pair_state=pair_state,
                    distance=distance,
                    current_time=current_time,
                )
                self._update_approach_state(
                    moments=moments,
                    pair_state=pair_state,
                    current_time=current_time,
                )

            for pair_key, pair_state in pair_states.items():
                if pair_key in seen_pairs:
                    continue
                pair_state.missing_count += 1
                self._update_pair_when_missing(
                    moments=moments,
                    pair_state=pair_state,
                    current_time=current_time,
                )

        video_end_time = frame_times[last_frame]
        self._close_open_states(moments, tracks, pair_states, video_end_time)
        return _merge_moments(moments, self.config.merge_gap_sec)

    def _update_stop_state(
        self,
        moments: list[Moment],
        state: _TrackState,
        current_time: float,
        track_id: TrackId,
    ) -> None:
        if state.ema_speed is None:
            state.below_count = 0
            state.below_start_time = None
            state.above_count = 0
            state.above_start_time = None
            return

        if state.ema_speed < self.config.stop_speed_threshold:
            if state.below_count == 0:
                state.below_start_time = current_time
            state.below_count += 1
        else:
            state.below_count = 0
            state.below_start_time = None

        if state.ema_speed > self.config.movement_speed_threshold:
            if state.above_count == 0:
                state.above_start_time = current_time
            state.above_count += 1
        else:
            state.above_count = 0
            state.above_start_time = None

        if state.active_stop_start is None:
            if state.below_count >= self.config.stop_enter_frames and state.below_start_time is not None:
                state.active_stop_start = state.below_start_time
                state.min_stop_speed = state.ema_speed
        else:
            if state.min_stop_speed is None:
                state.min_stop_speed = state.ema_speed
            else:
                state.min_stop_speed = min(state.min_stop_speed, state.ema_speed)
            if state.above_count >= self.config.stop_exit_frames and state.above_start_time is not None:
                _append_moment(
                    moments,
                    "STOP",
                    state.active_stop_start,
                    state.above_start_time,
                    [track_id],
                    {
                        "label": state.label,
                        "label_group": state.label_group,
                        "min_speed": state.min_stop_speed,
                    },
                )
                state.active_stop_start = None
                state.min_stop_speed = None
                state.below_count = 0
                state.below_start_time = None
                state.above_count = 0
                state.above_start_time = None

    def _update_near_state(
        self,
        moments: list[Moment],
        pair_state: _PairState,
        distance: float,
        current_time: float,
    ) -> None:
        if distance < self.config.near_threshold:
            if pair_state.near_count == 0:
                pair_state.near_start_time = current_time
            pair_state.near_count += 1
        else:
            pair_state.near_count = 0
            pair_state.near_start_time = None

        if distance > self.config.near_threshold_exit:
            if pair_state.far_count == 0:
                pair_state.far_start_time = current_time
            pair_state.far_count += 1
        else:
            pair_state.far_count = 0
            pair_state.far_start_time = None

        if pair_state.near_active_start is None:
            if pair_state.near_count >= self.config.near_enter_frames and pair_state.near_start_time is not None:
                pair_state.near_active_start = pair_state.near_start_time
                pair_state.near_min_distance = distance
        else:
            if pair_state.near_min_distance is None:
                pair_state.near_min_distance = distance
            else:
                pair_state.near_min_distance = min(pair_state.near_min_distance, distance)
            if pair_state.far_count >= self.config.near_exit_frames and pair_state.far_start_time is not None:
                _append_moment(
                    moments,
                    "NEAR",
                    pair_state.near_active_start,
                    pair_state.far_start_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "min_distance": pair_state.near_min_distance,
                    },
                )
                pair_state.near_active_start = None
                pair_state.near_count = 0
                pair_state.near_start_time = None
                pair_state.near_min_distance = None

    def _update_approach_state(
        self,
        moments: list[Moment],
        pair_state: _PairState,
        current_time: float,
    ) -> None:
        distances = list(pair_state.distances)
        if len(distances) >= 2:
            if distances[-1] > distances[-2]:
                if pair_state.approach_reverse_count == 0:
                    pair_state.approach_reverse_start_time = current_time
                pair_state.approach_reverse_count += 1
            else:
                pair_state.approach_reverse_count = 0
                pair_state.approach_reverse_start_time = None

        if len(distances) == self.config.approach_window:
            drop = distances[0] - distances[-1]
            consistently_decreasing = _strictly_decreasing(distances)
            if (
                pair_state.approach_active_start is None
                and consistently_decreasing
                and drop >= self.config.approach_drop_threshold
            ):
                pair_state.approach_active_start = pair_state.times[0]
                pair_state.approach_start_distance = distances[0]
                pair_state.approach_min_distance = distances[-1]
                pair_state.approach_reverse_count = 0
                pair_state.approach_reverse_start_time = None

        if pair_state.approach_active_start is not None:
            latest = distances[-1] if distances else None
            if latest is not None:
                if pair_state.approach_min_distance is None:
                    pair_state.approach_min_distance = latest
                else:
                    pair_state.approach_min_distance = min(pair_state.approach_min_distance, latest)
            if (
                pair_state.approach_reverse_count >= self.config.approach_reverse_frames
                and pair_state.approach_reverse_start_time is not None
            ):
                start_distance = pair_state.approach_start_distance or 0.0
                min_distance = pair_state.approach_min_distance or start_distance
                _append_moment(
                    moments,
                    "APPROACH",
                    pair_state.approach_active_start,
                    pair_state.approach_reverse_start_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "distance_drop": max(0.0, start_distance - min_distance),
                    },
                )
                pair_state.approach_active_start = None
                pair_state.approach_start_distance = None
                pair_state.approach_min_distance = None
                pair_state.approach_reverse_count = 0
                pair_state.approach_reverse_start_time = None

    def _update_pair_when_missing(
        self,
        moments: list[Moment],
        pair_state: _PairState,
        current_time: float,
    ) -> None:
        if pair_state.near_active_start is not None:
            if pair_state.far_count == 0:
                pair_state.far_start_time = current_time
            pair_state.far_count += 1
            if pair_state.far_count >= self.config.near_exit_frames and pair_state.far_start_time is not None:
                _append_moment(
                    moments,
                    "NEAR",
                    pair_state.near_active_start,
                    pair_state.far_start_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "min_distance": pair_state.near_min_distance,
                    },
                )
                pair_state.near_active_start = None
                pair_state.near_count = 0
                pair_state.near_start_time = None
                pair_state.near_min_distance = None
                pair_state.far_count = 0
                pair_state.far_start_time = None
        else:
            pair_state.near_count = 0
            pair_state.near_start_time = None
            pair_state.far_count = 0
            pair_state.far_start_time = None

        if pair_state.approach_active_start is not None:
            if pair_state.approach_reverse_count == 0:
                pair_state.approach_reverse_start_time = current_time
            pair_state.approach_reverse_count += 1
            if (
                pair_state.approach_reverse_count >= self.config.approach_reverse_frames
                and pair_state.approach_reverse_start_time is not None
            ):
                start_distance = pair_state.approach_start_distance or 0.0
                min_distance = pair_state.approach_min_distance or start_distance
                _append_moment(
                    moments,
                    "APPROACH",
                    pair_state.approach_active_start,
                    pair_state.approach_reverse_start_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "distance_drop": max(0.0, start_distance - min_distance),
                    },
                )
                pair_state.approach_active_start = None
                pair_state.approach_start_distance = None
                pair_state.approach_min_distance = None
                pair_state.approach_reverse_count = 0
                pair_state.approach_reverse_start_time = None
        else:
            if pair_state.missing_count >= self.config.approach_window:
                pair_state.distances.clear()
                pair_state.times.clear()
            pair_state.approach_reverse_count = 0
            pair_state.approach_reverse_start_time = None

    def _close_open_states(
        self,
        moments: list[Moment],
        tracks: dict[TrackId, _TrackState],
        pair_states: dict[tuple[TrackId, TrackId], _PairState],
        video_end_time: float,
    ) -> None:
        for track_id, state in tracks.items():
            if state.active_stop_start is not None:
                _append_moment(
                    moments,
                    "STOP",
                    state.active_stop_start,
                    state.last_seen_time,
                    [track_id],
                    {
                        "label": state.label,
                        "label_group": state.label_group,
                        "min_speed": state.min_stop_speed,
                    },
                )
        for pair_state in pair_states.values():
            if pair_state.near_active_start is not None:
                _append_moment(
                    moments,
                    "NEAR",
                    pair_state.near_active_start,
                    video_end_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "min_distance": pair_state.near_min_distance,
                    },
                )
            if pair_state.approach_active_start is not None:
                start_distance = pair_state.approach_start_distance or 0.0
                min_distance = pair_state.approach_min_distance or start_distance
                _append_moment(
                    moments,
                    "APPROACH",
                    pair_state.approach_active_start,
                    video_end_time,
                    list(pair_state.entities),
                    {
                        "class_pair": list(pair_state.class_pair),
                        "distance_drop": max(0.0, start_distance - min_distance),
                    },
                )


def generate_moments(
    observations: Iterable[TrackObservation | Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
    config: MomentConfig | None = None,
) -> list[Moment]:
    return MomentGenerator(config=config).generate(
        observations=observations,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _parse_observations(
    observations: Iterable[TrackObservation | Mapping[str, Any]],
) -> list[TrackObservation]:
    parsed: list[TrackObservation] = []
    for item in observations:
        if isinstance(item, TrackObservation):
            parsed.append(item)
        else:
            parsed.append(TrackObservation.from_mapping(item))

    dedup: dict[tuple[int, TrackId], TrackObservation] = {}
    for obs in parsed:
        key = (obs.frame_idx, obs.track_id)
        prev = dedup.get(key)
        if prev is None or obs.confidence > prev.confidence:
            dedup[key] = obs

    out = sorted(dedup.values(), key=lambda item: (item.frame_idx, item.time_sec, str(item.track_id)))
    return out


def _build_frame_index(
    observations: list[TrackObservation],
) -> tuple[dict[int, dict[TrackId, TrackObservation]], dict[int, float]]:
    frame_map: dict[int, dict[TrackId, TrackObservation]] = {}
    frame_times_raw: dict[int, list[float]] = {}
    for obs in observations:
        frame_map.setdefault(obs.frame_idx, {})[obs.track_id] = obs
        frame_times_raw.setdefault(obs.frame_idx, []).append(obs.time_sec)

    frame_anchor_times = {frame: statistics.mean(times) for frame, times in frame_times_raw.items()}
    sorted_anchors = sorted(frame_anchor_times.items())
    sec_per_frame = _estimate_seconds_per_frame(sorted_anchors)

    first_frame = sorted_anchors[0][0]
    last_frame = sorted_anchors[-1][0]
    frame_times: dict[int, float] = {}
    prev_time: float | None = None
    for frame_idx in range(first_frame, last_frame + 1):
        if frame_idx in frame_anchor_times:
            current = frame_anchor_times[frame_idx]
            if prev_time is not None and current <= prev_time:
                current = prev_time + sec_per_frame
        else:
            if prev_time is None:
                current = frame_idx * sec_per_frame
            else:
                current = prev_time + sec_per_frame
        frame_times[frame_idx] = current
        prev_time = current
    return frame_map, frame_times


def _estimate_seconds_per_frame(anchors: list[tuple[int, float]]) -> float:
    ratios: list[float] = []
    for idx in range(1, len(anchors)):
        frame_delta = anchors[idx][0] - anchors[idx - 1][0]
        time_delta = anchors[idx][1] - anchors[idx - 1][1]
        if frame_delta > 0 and time_delta > 0:
            ratios.append(time_delta / frame_delta)
    if ratios:
        return max(1e-6, statistics.median(ratios))
    return 1.0 / 30.0


def _class_group(label: str) -> str:
    value = label.lower().replace("_", " ").strip()
    if any(token in value for token in ("person", "pedestrian", "human", "walker")):
        return "person"
    if any(token in value for token in ("truck", "lorry", "semi", "pickup")):
        return "truck"
    if any(
        token in value
        for token in ("car", "vehicle", "automobile", "sedan", "suv", "van", "taxi")
    ):
        return "car"
    tokenized = value.split()
    return tokenized[0] if tokenized else "unknown"


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _strictly_decreasing(values: list[float], eps: float = 1e-6) -> bool:
    return all(values[idx + 1] < values[idx] - eps for idx in range(len(values) - 1))


def _append_moment(
    moments: list[Moment],
    moment_type: str,
    start_time: float,
    end_time: float,
    entities: list[TrackId],
    metadata: dict[str, Any],
) -> None:
    start = float(start_time)
    end = float(end_time)
    if end < start:
        end = start
    moments.append(
        Moment(
            type=moment_type,
            start_time=start,
            end_time=end,
            entities=sorted(entities, key=str),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
    )


def _merge_moments(moments: list[Moment], max_gap_sec: float) -> list[Moment]:
    if not moments:
        return []

    grouped = sorted(moments, key=lambda m: (m.type, tuple(m.entities), m.start_time, m.end_time))
    merged: list[Moment] = []

    for moment in grouped:
        if not merged:
            merged.append(moment)
            continue
        prev = merged[-1]
        same_key = prev.type == moment.type and tuple(prev.entities) == tuple(moment.entities)
        adjacent = moment.start_time <= prev.end_time + max_gap_sec
        if same_key and adjacent:
            prev.end_time = max(prev.end_time, moment.end_time)
            prev.metadata = _merge_metadata(prev.metadata, moment.metadata)
        else:
            merged.append(moment)

    return sorted(merged, key=lambda m: (m.start_time, m.end_time, m.type))


def _merge_metadata(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in other.items():
        if key not in out:
            out[key] = value
            continue
        existing = out[key]
        if isinstance(existing, (int, float)) and isinstance(value, (int, float)):
            if "min" in key:
                out[key] = min(existing, value)
            elif "max" in key or "drop" in key:
                out[key] = max(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            out[key] = sorted({*existing, *value})
    return out
