from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class VideoTranscript:
    video_id: str
    title: str
    domain: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: list[TranscriptSegment] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "VideoTranscript":
        required = ("video_id", "title", "domain", "segments")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

        segments: list[TranscriptSegment] = []
        for raw in data["segments"]:
            if not raw.get("text", "").strip():
                continue
            segments.append(
                TranscriptSegment(
                    start=float(raw.get("start", 0.0)),
                    end=float(raw.get("end", 0.0)),
                    text=str(raw["text"]).strip(),
                )
            )

        if not segments:
            raise ValueError("Transcript has no usable segments")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")

        return VideoTranscript(
            video_id=str(data["video_id"]),
            title=str(data["title"]),
            domain=str(data["domain"]),
            url=data.get("url"),
            metadata=metadata,
            segments=segments,
        )


@dataclass(slots=True)
class VideoChunk:
    video_id: str
    title: str
    domain: str
    text: str
    start: float
    end: float
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

