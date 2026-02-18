from __future__ import annotations

from dataclasses import dataclass

from .types import TranscriptSegment, VideoChunk, VideoTranscript


@dataclass(slots=True)
class ChunkConfig:
    max_words: int = 140
    stride_words: int = 80
    min_words: int = 30

    def validate(self) -> None:
        if self.max_words <= 0 or self.stride_words <= 0 or self.min_words <= 0:
            raise ValueError("Chunk sizes must be positive")
        if self.stride_words >= self.max_words:
            raise ValueError("stride_words must be smaller than max_words")
        if self.min_words > self.max_words:
            raise ValueError("min_words cannot exceed max_words")


def _count_words(text: str) -> int:
    return len(text.split())


def _window_to_chunk(transcript: VideoTranscript, window: list[TranscriptSegment]) -> VideoChunk:
    return VideoChunk(
        video_id=transcript.video_id,
        title=transcript.title,
        domain=transcript.domain,
        text=" ".join(segment.text for segment in window).strip(),
        start=window[0].start,
        end=window[-1].end,
        url=transcript.url,
        metadata=transcript.metadata.copy(),
    )


def chunk_segments(transcript: VideoTranscript, config: ChunkConfig) -> list[VideoChunk]:
    """Chunk transcript segments into overlapping text windows."""
    config.validate()
    chunks: list[VideoChunk] = []
    window: list[TranscriptSegment] = []
    window_word_count = 0

    for segment in transcript.segments:
        words = _count_words(segment.text)
        if words == 0:
            continue
        window.append(segment)
        window_word_count += words

        while window_word_count >= config.max_words:
            chunks.append(_window_to_chunk(transcript, window))

            # Slide the window forward until we keep roughly stride_words words.
            while window and window_word_count > config.stride_words:
                popped = window.pop(0)
                window_word_count -= _count_words(popped.text)

    if window and window_word_count >= config.min_words:
        chunks.append(_window_to_chunk(transcript, window))

    return chunks

