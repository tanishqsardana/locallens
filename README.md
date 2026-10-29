# LocalLens

LocalLens turns a video into searchable, reviewable events. It samples frames, detects and tracks objects, derives moments such as `APPEAR`, `DISAPPEAR`, `STOP`, `NEAR`, and `APPROACH`, and stores the result in a local SQLite index.

It supports two related workflows:

- **Transcript search:** index timestamped transcript segments and retrieve them with natural-language queries.
- **Video moment search:** process video plus detections/tracks, generate visual moments, inspect keyframes, search the moment index, and export short verification clips.

## Demo

Drop in a video, detect and track objects, then search the generated moments.



https://github.com/user-attachments/assets/3df25c35-fcf7-4a8f-8716-7d72b34727d9



## Requirements

- Python 3.10+
- For video processing: FFmpeg and a supported OpenCV video backend
- For YOLO-World detection: a compatible PyTorch/Ultralytics runtime; GPU acceleration is optional but recommended
- For generated captions or vocabulary refinement: an OpenAI-compatible vision-language endpoint

The default video configuration targets a CUDA machine and a local VLM endpoint. On a CPU or Apple Silicon machine, change **Detector device** in the app or pass the appropriate CLI option (for example, `--yoloworld-device mps`).

## Install

Install the base package:

```bash
python -m pip install -e .
```

Install the web application and video-processing dependencies:

```bash
python -m pip install -e ".[app,video]"
```

Optional: use a sentence-transformer model rather than the lightweight hashing embedder for semantic retrieval:

```bash
python -m pip install -e ".[semantic]"
```

## Run the web app

```bash
streamlit run streamlit_app.py
```

Open **Pipeline Runner**, then drag a video onto the upload area (or click it to browse). LocalLens previews the selected file before processing it. It accepts MP4, MOV, MKV, AVI, WebM, and M4V files; the included Streamlit configuration raises the upload limit to 1 GB.

Choose **Run Full Pipeline** to create a run in `data/video_runs/<video-name>/`. The default pipeline uses YOLO-World and VLM captions. Expand **Advanced** to:

- select `cpu`, `cuda`, or `mps` for the detector;
- turn off **Generate VLM captions** when no vision-language endpoint is available;
- turn off vocabulary refinement independently;
- select `hashing` or `sentence-transformer` for the searchable text index.

After a run, use **Semantic Query** to find the top three matching moments and generate playable clips. **Cycle Inspector** reopens an earlier run; **Moment Lab** is for experimenting with tracking JSON and moment thresholds.

## Video pipeline

```text
video
  -> sampled frames
  -> optional VLM captions and vocabulary
  -> detections and tracking
  -> normalized tracks
  -> moments
  -> keyframes and embeddings
  -> SQLite semantic index
  -> search and verification clips
```

The pipeline accepts exactly one tracking source:

| Source | CLI option | Use it when |
| --- | --- | --- |
| Existing tracks | `--tracks tracks.json` | A tracker has already run. |
| ByteTrack MOT output | `--bytetrack-txt results.txt` | You have standard MOT-format output. |
| Frame detections | `--detections detections.json` | LocalLens should perform simple IoU tracking. |
| YOLO-World | `--auto-detections-yoloworld` | LocalLens should detect and track objects. |
| GroundingDINO | `--auto-detections-groundingdino` | You have the GroundingDINO config and weights. |

### Run from the command line

The repository includes `config/video_cycle.defaults.json`, which is automatically applied unless disabled. It enables YOLO-World, VLM captions, and VLM vocabulary refinement.

With the default stack configured:

```bash
python -m videosearch.video_cycle_cli \
  --video /path/to/video.mp4 \
  --out-dir data/video_cycle_run \
  --show-phase-outputs
```

To process an existing tracking export without requiring a VLM endpoint, opt out of the defaults:

```bash
python -m videosearch.video_cycle_cli \
  --no-default-config \
  --video /path/to/video.mp4 \
  --tracks /path/to/tracks.json \
  --out-dir data/video_cycle_run
```

To run YOLO-World with no captions (for example, on Apple Silicon):

```bash
python -m videosearch.video_cycle_cli \
  --no-default-config \
  --video /path/to/video.mp4 \
  --auto-detections-yoloworld \
  --yoloworld-device mps \
  --out-dir data/video_cycle_run
```

Pass `--config path/to/config.json` to use another settings file. Explicit CLI options override configuration values.

### Input data

Track JSON may be either a list of rows or an object with an `observations` array. Each row needs equivalent fields for:

```json
{
  "track_id": 7,
  "class": "car",
  "bbox": [100, 200, 180, 260],
  "confidence": 0.92,
  "frame_idx": 42,
  "time_sec": 1.4
}
```

Accepted aliases include `id` for `track_id`, `label` for `class`, `xyxy` for `bbox`, `score` for `confidence`, `frame` for `frame_idx`, and `timestamp` for `time_sec`.

### Run artifacts

Each run writes its source inputs and derived files to its output directory. The most useful artifacts are:

- `ingest/video_manifest.json` and `ingest/sampled_frames.json`
- `normalized_tracks.json` and `tracks_report.json`
- `moments.json` and `moment_keyframes.json`
- `moment_index.sqlite`
- `phase_outputs.json` for inspection in the web app

VLM-enabled runs can also produce `vlm_captions_generated.json`, `vocabulary.json`, and `vocab_postprocess.json`. Detector-driven runs store generated detections and `tracked_rows.json`.

## Search moments and export clips

Search a completed video run from the command line:

```bash
python -m videosearch.moment_semantic_cli search \
  --run-dir data/video_cycle_run \
  --query "truck enters the scene" \
  --top-k 8
```

Query structured events directly:

```bash
python -m videosearch.moment_query_cli appear \
  --run-dir data/video_cycle_run \
  --label truck
```

Render a video with moment overlays:

```bash
python -m videosearch.moment_overlay_cli \
  --video /path/to/video.mp4 \
  --run-dir data/video_cycle_run \
  --out-video data/video_cycle_run/moment_overlay.mp4
```

Export clips for appearances of one label:

```bash
python -m videosearch.moment_clip_cli \
  --video /path/to/video.mp4 \
  --run-dir data/video_cycle_run \
  --label truck \
  --out-dir data/video_cycle_run/truck_episode_clips
```

## Transcript search

Index the provided examples with the dependency-free hashing embedder:

```bash
python -m videosearch.cli index \
  --input-dir examples/transcripts \
  --db data/video_index.sqlite \
  --policy config/domain_chunking.json \
  --embedder hashing \
  --recreate
```

Search the index:

```bash
python -m videosearch.cli search \
  --db data/video_index.sqlite \
  --query "How do I reduce risk in a volatile portfolio?" \
  --domain finance \
  --top-k 3 \
  --embedder hashing
```

Transcript JSON must include a video identifier, metadata, and timestamped `segments`:

```json
{
  "video_id": "fin-001",
  "title": "Portfolio Basics for Volatile Markets",
  "domain": "finance",
  "url": "https://example.com/videos/fin-001",
  "metadata": {"speaker": "Analyst Team"},
  "segments": [{"start": 0.0, "end": 11.2, "text": "..."}]
}
```

## Project layout

| Path | Purpose |
| --- | --- |
| `streamlit_app.py` | Upload, inspect, query, and clip-export interface. |
| `src/videosearch/video_cycle.py` | End-to-end video pipeline. |
| `src/videosearch/moments.py` | Rule-based moment generation from tracks. |
| `src/videosearch/*_cli.py` | Pipeline, query, clip, overlay, and semantic CLI commands. |
| `src/videosearch/pipeline.py` | Transcript indexing and retrieval orchestration. |
| `config/` | Domain chunking and video pipeline defaults. |
| `examples/` | Transcript, track, caption, and synonym examples. |
| `tests/` | Unit tests for indexing, moments, query, overlay, clips, and video helpers. |

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```
