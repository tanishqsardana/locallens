from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunking import ChunkConfig, chunk_segments
from .embedders import BaseEmbedder
from .store import SearchResult, SqliteVectorStore
from .types import VideoChunk, VideoTranscript


@dataclass(slots=True)
class DomainChunkingPolicy:
    default: ChunkConfig
    per_domain: dict[str, ChunkConfig]

    def for_domain(self, domain: str) -> ChunkConfig:
        return self.per_domain.get(domain, self.default)

    @staticmethod
    def from_dict(data: dict[str, object]) -> "DomainChunkingPolicy":
        def parse_chunk(values: dict[str, object]) -> ChunkConfig:
            config = ChunkConfig(
                max_words=int(values.get("max_words", 140)),
                stride_words=int(values.get("stride_words", 80)),
                min_words=int(values.get("min_words", 30)),
            )
            config.validate()
            return config

        default_cfg = parse_chunk(data.get("default", {}) if isinstance(data.get("default", {}), dict) else {})
        raw_domains = data.get("domains", {})
        domain_cfgs: dict[str, ChunkConfig] = {}
        if isinstance(raw_domains, dict):
            for domain, values in raw_domains.items():
                if isinstance(values, dict):
                    domain_cfgs[str(domain)] = parse_chunk(values)
        return DomainChunkingPolicy(default=default_cfg, per_domain=domain_cfgs)

    @staticmethod
    def default_policy() -> "DomainChunkingPolicy":
        return DomainChunkingPolicy(default=ChunkConfig(), per_domain={})


class VideoSemanticSearch:
    def __init__(self, store: SqliteVectorStore, embedder: BaseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def index_transcripts(
        self,
        transcripts: Iterable[VideoTranscript],
        policy: DomainChunkingPolicy | None = None,
        batch_size: int = 64,
    ) -> int:
        policy = policy or DomainChunkingPolicy.default_policy()
        chunks: list[VideoChunk] = []
        for transcript in transcripts:
            config = policy.for_domain(transcript.domain)
            chunks.extend(chunk_segments(transcript, config))

        if not chunks:
            return 0

        self.store.set_metadata("model_name", self.embedder.model_name)
        self.store.set_metadata("embedding_dim", str(self.embedder.embedding_dim))

        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            vectors = self.embedder.embed([chunk.text for chunk in batch])
            if vectors.shape[1] != self.embedder.embedding_dim:
                raise ValueError("Embedder produced unexpected vector dimension")
            self.store.upsert_chunks(
                chunks=batch,
                embeddings=vectors,
                model_name=self.embedder.model_name,
                embedding_dim=self.embedder.embedding_dim,
            )
        return len(chunks)

    def search(self, query: str, top_k: int = 5, domain: str | None = None) -> list[SearchResult]:
        if not query.strip():
            return []
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(
            query_embedding=query_vec,
            top_k=top_k,
            domain=domain,
            model_name=self.embedder.model_name,
        )


def load_transcripts(input_dir: str | Path) -> list[VideoTranscript]:
    path = Path(input_dir)
    transcripts: list[VideoTranscript] = []
    for file in sorted(path.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        transcripts.append(VideoTranscript.from_dict(data))
    return transcripts


def load_policy(policy_path: str | Path | None) -> DomainChunkingPolicy:
    if policy_path is None:
        return DomainChunkingPolicy.default_policy()
    raw = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Policy file must be a JSON object")
    return DomainChunkingPolicy.from_dict(raw)


def ensure_model_compatibility(store: SqliteVectorStore, model_name: str, embedding_dim: int) -> None:
    stored_model = store.get_metadata("model_name")
    stored_dim = store.get_metadata("embedding_dim")
    if stored_model and stored_model != model_name:
        raise ValueError(
            f"Index built with model '{stored_model}', but current embedder is '{model_name}'. "
            "Use --recreate to rebuild the index."
        )
    if stored_dim and int(stored_dim) != embedding_dim:
        raise ValueError(
            f"Index built with embedding dim {stored_dim}, but current embedder uses {embedding_dim}. "
            "Use --recreate to rebuild the index."
        )


def to_json(results: list[SearchResult]) -> str:
    payload = [
        {
            "score": round(item.score, 4),
            "video_id": item.chunk.video_id,
            "title": item.chunk.title,
            "domain": item.chunk.domain,
            "time_range": [item.chunk.start, item.chunk.end],
            "url": item.chunk.url,
            "text": item.chunk.text,
            "metadata": item.chunk.metadata,
        }
        for item in results
    ]
    return json.dumps(payload, indent=2, ensure_ascii=True)

