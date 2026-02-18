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

The app now has two views in the sidebar:
- `Cycle Inspector`: load and inspect `phase_outputs.json` with phase tabs and keyframe previews
- `Moment Lab`: run moment generation directly from tracking rows

## Full Video Cycle (Video -> Moments -> Keyframes -> Index)

Install video dependencies:

```bash
python -m pip install -e ".[video]"
```

Run the full cycle:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --tracks /path/to/tracks.json \
  --out-dir data/video_cycle_run \
  --captions /path/to/vlm_captions.json \
  --synonyms /path/to/synonyms.json \
  --seed-labels car,truck,person \
  --target-fps 10 \
  --show-phase-outputs
```

Or run tracking in-pipeline from frame-level detections:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --detections /path/to/detections.json \
  --out-dir data/video_cycle_run \
  --detect-track-iou-threshold 0.3 \
  --detect-track-max-missed-frames 10 \
  --show-phase-outputs
```

Run YOLO-World detections + tracking in one pipeline command:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --auto-detections-yoloworld \
  --yoloworld-model yolov8s-worldv2.pt \
  --yoloworld-confidence 0.2 \
  --yoloworld-device cuda \
  --out-dir data/video_cycle_run \
  --auto-captions \
  --vlm-endpoint http://localhost:8000/v1/chat/completions \
  --vlm-model nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
  --llm-postprocess-vocab \
  --detect-track-iou-threshold 0.3 \
  --detect-track-max-missed-frames 10 \
  --log-progress \
  --show-phase-outputs
```

Run GroundingDINO detections + tracking in one pipeline command:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --auto-detections-groundingdino \
  --groundingdino-config-path /path/to/GroundingDINO_SwinT_OGC.py \
  --groundingdino-weights-path /path/to/groundingdino_swint_ogc.pth \
  --out-dir data/video_cycle_run \
  --auto-captions \
  --vlm-endpoint http://localhost:8000/v1/chat/completions \
  --vlm-model nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
  --llm-postprocess-vocab \
  --detect-track-iou-threshold 0.3 \
  --detect-track-max-missed-frames 10 \
  --show-phase-outputs
```

Phase 3 track processing controls:
- `--track-min-confidence` (drop low-confidence rows)
- `--track-min-length` (drop short tracks)
- `--track-max-interp-gap` (optional interpolation for short missing gaps)
- `--track-no-clip-bbox` (disable bbox clipping to frame bounds)
- `--moment-labels` (default: `car,truck,bus,van,person,motorcycle`; filters out static/background classes)

Auto-generate captions from a local vLLM VLM endpoint (OpenAI-compatible):

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --bytetrack-txt /path/to/bytetrack_results.txt \
  --out-dir data/video_cycle_run \
  --auto-captions \
  --vlm-endpoint http://localhost:8000/v1/chat/completions \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --vlm-frame-stride 10 \
  --vlm-max-tokens 120 \
  --llm-postprocess-vocab \
  --llm-postprocess-max-detection-terms 20 \
  --show-phase-outputs
```

If you already ran ByteTrack and have MOT txt output:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --bytetrack-txt /path/to/bytetrack_results.txt \
  --bytetrack-class car \
  --out-dir data/video_cycle_run \
  --show-phase-outputs
```

Pipeline artifacts:
- `data/video_cycle_run/ingest/video_manifest.json`
- `data/video_cycle_run/ingest/sampled_frames.json`
- `data/video_cycle_run/vocabulary.json` (if captions are provided)
- `data/video_cycle_run/vlm_captions_generated.json` (if `--auto-captions` is used)
- `data/video_cycle_run/vocab_postprocess.json` (if `--llm-postprocess-vocab` is used)
- `data/video_cycle_run/tracked_rows.json` (if `--detections` is used)
- `data/video_cycle_run/yoloworld_detections_generated.json` (if `--auto-detections-yoloworld` is used)
- `data/video_cycle_run/groundingdino_detections_generated.json` (if `--auto-detections-groundingdino` is used)
- `data/video_cycle_run/normalized_tracks.json`
- `data/video_cycle_run/tracks_report.json`
- `data/video_cycle_run/moments.json`
- `data/video_cycle_run/moment_keyframes.json`
- `data/video_cycle_run/moment_index.sqlite`
- `data/video_cycle_run/phase_outputs.json` (phase-by-phase debug payload)

For full untruncated phase payloads during development:

```bash
PYTHONPATH=src python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --tracks /path/to/tracks.json \
  --out-dir data/video_cycle_run \
  --show-full-phase-outputs
```

Track JSON input can be either:
- list of rows, or
- object with `observations` list.

Each row must contain equivalents of:
- `track_id` (or `id`)
- `class` (or `label` / `class_name` / `cls`)
- `bbox` `[x1,y1,x2,y2]` (or `xyxy`)
- `confidence` (or `score` / `conf`)
- `frame_idx` (or `frame` / `frame_id`)
- `time_sec` (or `timestamp` / `time`)

`--tracks`, `--bytetrack-txt`, `--detections`, `--auto-detections-groundingdino`, and `--auto-detections-yoloworld` are mutually exclusive.
Provide exactly one.
`--captions` and `--auto-captions` are mutually exclusive.

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
- `TRAFFIC_CHANGE`

`DISAPPEAR` is emitted as a point event once missing-frame threshold is reached.
`APPEAR` continuity tolerates short frame gaps (useful when ingest FPS is lower than source FPS).

All distances are normalized by frame diagonal. Speed uses EMA smoothing.
Overlapping/adjacent moments with gap `< 1s` are merged.

## Operational Query Layer

Query the generated moment/track artifacts for questions like:
- "when does truck appear?"
- "which frames contain truck?"
- "when does this car pass through?"

```bash
PYTHONPATH=src python -m videosearch.moment_query_cli appear \
  --run-dir data/video_cycle_run \
  --label truck
```

```bash
PYTHONPATH=src python -m videosearch.moment_query_cli frames-with \
  --run-dir data/video_cycle_run \
  --label truck
```

```bash
PYTHONPATH=src python -m videosearch.moment_query_cli pass-through \
  --run-dir data/video_cycle_run \
  --label car \
  --frame-width 3840 \
  --frame-height 2160
```

Simple NLQ router:

```bash
PYTHONPATH=src python -m videosearch.moment_query_cli nlq \
  --run-dir data/video_cycle_run \
  --query "when does truck appear?" \
  --frame-width 3840 \
  --frame-height 2160
```

## Moment Overlay Video (Verification UI)

Render an MP4 overlay to visually verify moments against tracks:

```bash
PYTHONPATH=src python -m videosearch.moment_overlay_cli \
  --video /path/to/video.mp4 \
  --run-dir data/video_cycle_run \
  --out-video data/video_cycle_run/moment_overlay.mp4 \
  --log-every-frames 60
```

Useful flags:
- `--show-all-tracks` (draw every track, not only active moment entities)
- `--start-sec` / `--end-sec` (render only a time slice)
