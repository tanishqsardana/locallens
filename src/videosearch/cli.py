from __future__ import annotations

import argparse
from pathlib import Path

from .embedders import build_embedder
from .pipeline import (
    VideoSemanticSearch,
    ensure_model_compatibility,
    load_policy,
    load_transcripts,
    to_json,
)
from .store import SqliteVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain-adaptable semantic search for videos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Index transcripts into a vector store")
    index.add_argument("--input-dir", default="examples/transcripts", help="Directory with transcript JSON files")
    index.add_argument("--db", default="data/video_index.sqlite", help="SQLite index path")
    index.add_argument("--policy", default="config/domain_chunking.json", help="Chunking policy JSON path")
    index.add_argument("--embedder", default="hashing", choices=["hashing", "sentence-transformer"])
    index.add_argument("--model", default=None, help="Embedder model name if applicable")
    index.add_argument("--batch-size", type=int, default=64)
    index.add_argument("--recreate", action="store_true", help="Drop and rebuild the index")

    search = subparsers.add_parser("search", help="Search indexed video chunks")
    search.add_argument("--db", default="data/video_index.sqlite", help="SQLite index path")
    search.add_argument("--query", required=True, help="Natural-language query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--domain", default=None, help="Optional domain filter")
    search.add_argument("--embedder", default="hashing", choices=["hashing", "sentence-transformer"])
    search.add_argument("--model", default=None, help="Embedder model name if applicable")

    return parser


def cmd_index(args: argparse.Namespace) -> int:
    transcripts = load_transcripts(args.input_dir)
    policy = load_policy(args.policy if Path(args.policy).exists() else None)
    embedder = build_embedder(args.embedder, args.model)
    store = SqliteVectorStore(args.db)
    store.initialize(recreate=args.recreate)
    ensure_model_compatibility(store, embedder.model_name, embedder.embedding_dim)

    app = VideoSemanticSearch(store=store, embedder=embedder)
    count = app.index_transcripts(transcripts=transcripts, policy=policy, batch_size=args.batch_size)
    print(f"Indexed {count} chunks into {args.db} using {embedder.model_name}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    embedder = build_embedder(args.embedder, args.model)
    store = SqliteVectorStore(args.db)
    store.initialize(recreate=False)
    ensure_model_compatibility(store, embedder.model_name, embedder.embedding_dim)
    app = VideoSemanticSearch(store=store, embedder=embedder)
    results = app.search(query=args.query, top_k=args.top_k, domain=args.domain)
    print(to_json(results))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "index":
        return cmd_index(args)
    if args.command == "search":
        return cmd_search(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

