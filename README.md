# Quiet Observer

Local computer vision app: capture frames from YouTube live streams, label them, train a YOLO model, and run continuous inference — all on your Mac.

## Prerequisites

```bash
brew install ffmpeg yt-dlp
```

[uv](https://docs.astral.sh/uv/getting-started/installation/) is required for dependency management.

## Setup

```bash
uv sync
```

## Start

```bash
uv run uvicorn quiet_observer.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Workflow

1. **Create project** — paste a YouTube URL, set capture and inference intervals
2. **Capture** — start the capture worker on the project page; frames appear every N seconds
3. **Label** — draw bounding boxes, assign classes
4. **Train** — one click fine-tunes a YOLO model on your labeled frames (runs on MPS on Apple Silicon)
5. **Deploy** — pick a model version to deploy
6. **Monitor** — inference runs continuously; uncertain frames are queued for review

## Data

All frames, model weights, and logs are stored under `data/` (gitignored).
