from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import VideoChunk


@dataclass(slots=True)
class SearchResult:
    score: float
    chunk: VideoChunk


class SqliteVectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self, recreate: bool = False) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            if recreate:
                conn.execute("DROP TABLE IF EXISTS chunks;")
                conn.execute("DROP TABLE IF EXISTS index_metadata;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    url TEXT,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    model_name TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def set_metadata(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO index_metadata(key, value) VALUES(?, ?)",
                (key, value),
            )
            conn.commit()

    def get_metadata(self, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM index_metadata WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def upsert_chunks(
        self,
        chunks: Iterable[VideoChunk],
        embeddings: np.ndarray,
        model_name: str,
        embedding_dim: int,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = []
            for chunk, vector in zip(chunks, embeddings, strict=True):
                rows.append(
                    (
                        chunk.video_id,
                        chunk.title,
                        chunk.domain,
                        chunk.text,
                        chunk.start,
                        chunk.end,
                        chunk.url,
                        json.dumps(chunk.metadata, ensure_ascii=True, sort_keys=True),
                        np.asarray(vector, dtype=np.float32).tobytes(),
                        embedding_dim,
                        model_name,
                    )
                )
            conn.executemany(
                """
                INSERT INTO chunks(
                    video_id, title, domain, text, start, end, url, metadata_json,
                    embedding, embedding_dim, model_name
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        domain: str | None = None,
        model_name: str | None = None,
    ) -> list[SearchResult]:
        query = """
            SELECT
                video_id, title, domain, text, start, end, url, metadata_json,
                embedding, embedding_dim, model_name
            FROM chunks
            WHERE 1=1
        """
        params: list[object] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return []

        embeddings = np.vstack(
            [np.frombuffer(row[8], dtype=np.float32, count=row[9]) for row in rows]
        ).astype(np.float32)
        scores = embeddings @ query_embedding.astype(np.float32)
        top_idx = np.argsort(-scores)[:top_k]

        results: list[SearchResult] = []
        for idx in top_idx:
            row = rows[int(idx)]
            chunk = VideoChunk(
                video_id=row[0],
                title=row[1],
                domain=row[2],
                text=row[3],
                start=float(row[4]),
                end=float(row[5]),
                url=row[6],
                metadata=json.loads(row[7]),
            )
            results.append(SearchResult(score=float(scores[idx]), chunk=chunk))
        return results

