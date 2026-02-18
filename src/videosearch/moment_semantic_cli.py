from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from urllib import error as url_error
from urllib import request as url_request

from .video_cycle import extract_chat_completion_text, semantic_search_moments


def _resolve_db(run_dir: str | None, explicit_db: str | None) -> str:
    if explicit_db:
        return explicit_db
    if run_dir:
        return str(Path(run_dir) / "moment_index.sqlite")
    raise ValueError("Provide --db or --run-dir")


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
    req = url_request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with url_request.urlopen(req, timeout=float(timeout_sec)) as response:
            raw = response.read().decode("utf-8")
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return json.loads(raw)


def _build_llm_grounded_prompt(query: str, results: list[dict[str, Any]]) -> str:
    compact = []
    for row in results:
        compact.append(
            {
                "moment_index": row.get("moment_index"),
                "type": row.get("type"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "metadata": row.get("metadata", {}),
                "score": round(float(row.get("score", 0.0)), 4),
            }
        )
    return (
        "Answer the user question using only the retrieved moment records.\n"
        "If evidence is insufficient, say so clearly.\n"
        "Return concise JSON with keys: answer, confidence, cited_moment_indices.\n"
        f"Question: {query}\n"
        f"Retrieved moments: {json.dumps(compact, ensure_ascii=True)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search and grounded LLM QA over moment index")
    parser.add_argument("--run-dir", default=None, help="Directory containing moment_index.sqlite")
    parser.add_argument("--db", default=None, help="Path to moment_index.sqlite")

    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Semantic search over indexed moments")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)

    ask = sub.add_parser("ask", help="LLM-grounded answer using semantic retrieval context")
    ask.add_argument("--query", required=True)
    ask.add_argument("--top-k", type=int, default=12)
    ask.add_argument("--llm-endpoint", required=True)
    ask.add_argument("--llm-model", required=True)
    ask.add_argument("--llm-max-tokens", type=int, default=220)
    ask.add_argument("--llm-timeout-sec", type=float, default=60.0)
    ask.add_argument("--llm-temperature", type=float, default=0.0)
    ask.add_argument("--llm-api-key", default=None)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = _resolve_db(args.run_dir, args.db)

    if args.command == "search":
        rows = semantic_search_moments(db_path=db_path, query=args.query, top_k=int(args.top_k))
        print(json.dumps({"query": args.query, "results": rows}, indent=2, ensure_ascii=True))
        return 0

    if args.command == "ask":
        rows = semantic_search_moments(db_path=db_path, query=args.query, top_k=int(args.top_k))
        prompt = _build_llm_grounded_prompt(args.query, rows)
        payload = {
            "model": args.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(args.llm_temperature),
            "max_tokens": int(args.llm_max_tokens),
        }
        response = _post_chat_completion(
            endpoint=args.llm_endpoint,
            payload=payload,
            timeout_sec=float(args.llm_timeout_sec),
            api_key=args.llm_api_key,
        )
        answer = extract_chat_completion_text(response)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "retrieved": rows,
                    "llm_answer": answer,
                    "llm_model": args.llm_model,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
