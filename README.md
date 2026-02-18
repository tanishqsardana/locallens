# locallens

Domain-adaptable semantic search over video transcripts.

This starter gives you:
- Transcript ingestion from JSON files
- Configurable chunking (global + per-domain overrides)
- Pluggable embedders (`sentence-transformer` for semantic quality, `hashing` fallback)
- SQLite vector index with cosine retrieval
- CLI for indexing and search

## Quick Start

```bash
python -m pip install -e .
```

Optional (better semantic quality):
```bash
python -m pip install -e ".[semantic]"
```

Index sample videos:
```bash
PYTHONPATH=src python -m videosearch.cli index \
  --input-dir examples/transcripts \
  --db data/video_index.sqlite \
  --policy config/domain_chunking.json \
  --embedder hashing \
  --recreate
```

Search:
```bash
PYTHONPATH=src python -m videosearch.cli search \
  --db data/video_index.sqlite \
  --query "How do I reduce risk in a volatile portfolio?" \
  --top-k 3 \
  --domain finance \
  --embedder hashing
```

To use transformer embeddings:
```bash
PYTHONPATH=src python -m videosearch.cli index \
  --embedder sentence-transformer \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --recreate
```

## Transcript Schema

Each input file in `examples/transcripts/*.json`:

```json
{
  "video_id": "fin-001",
  "title": "Portfolio Basics for Volatile Markets",
  "domain": "finance",
  "url": "https://example.com/videos/fin-001",
  "metadata": { "speaker": "Analyst Team" },
  "segments": [
    { "start": 0.0, "end": 11.2, "text": "..." }
  ]
}
```

`domain` is first-class and can be used for:
- Per-domain chunking policy
- Query-time filtering
- Routing to domain-specific rerankers (next step)

## Domain Adaptation Hooks

1. Chunking policy in `config/domain_chunking.json`:
   - `default` for all domains
   - `domains.<name>` overrides for specific verticals (medical, finance, etc.)
2. Embedder abstraction in `src/videosearch/embedders.py`:
   - Add new embedders without touching indexing/search code.
3. Storage abstraction in `src/videosearch/store.py`:
   - Swap SQLite for FAISS/Qdrant/Weaviate once scale grows.

## Project Layout

- `src/videosearch/cli.py`: CLI entrypoint (`index`, `search`)
- `src/videosearch/pipeline.py`: end-to-end indexing/search orchestration
- `src/videosearch/moments.py`: state-transition moment generation from tracked objects
- `src/videosearch/chunking.py`: transcript chunking with overlap
- `src/videosearch/embedders.py`: embedder interface + implementations
- `src/videosearch/store.py`: SQLite vector store and cosine retrieval
- `examples/transcripts/`: sample videos across domains
- `tests/test_pipeline.py`: basic index + query test
- `tests/test_moments.py`: moment generation tests for APPEAR/DISAPPEAR/STOP/NEAR/APPROACH

## Run Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

Current tests:
- `tests/test_pipeline.py::VideoSemanticSearchTest::test_index_and_search`
- `tests/test_moments.py::MomentGenerationTest::test_appear_and_disappear`
- `tests/test_moments.py::MomentGenerationTest::test_stop_event_from_speed_transition`
- `tests/test_moments.py::MomentGenerationTest::test_near_moments_are_merged_when_gap_is_small`
- `tests/test_moments.py::MomentGenerationTest::test_approach_event_from_decreasing_distance`

## Streamlit Moment Lab

Install app dependencies:

```bash
python -m pip install -e ".[app]"
```

Run:

```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

The app supports:
- Built-in sample (`examples/tracks/sample_static_scene.json`)
- Uploading your own JSON tracking outputs
- Interactive threshold tuning for APPEAR/DISAPPEAR/STOP/NEAR/APPROACH
- Exporting generated moments as JSON

## Moment Generation (Tracking Output -> Events)

`src/videosearch/moments.py` builds structured moments from tracking outputs only
(no VLM timestamping). Input per detection:
`track_id`, `class`, `bbox`, `confidence`, `frame_idx`, `time_sec`.

```python
from videosearch.moments import MomentConfig, generate_moments

rows = [
  {
    "track_id": 7,
    "class": "car",
    "bbox": [100, 200, 180, 260],
    "confidence": 0.92,
    "frame_idx": 42,
    "time_sec": 1.4
  }
]

moments = generate_moments(
  observations=rows,
  frame_width=1920,
  frame_height=1080,
  config=MomentConfig()
)
```

Returned moments:
- `APPEAR` and `DISAPPEAR`
- `STOP`
- `NEAR`
- `APPROACH`

All distances are normalized by frame diagonal. Speed uses EMA smoothing.
Overlapping/adjacent moments with gap `< 1s` are merged.
