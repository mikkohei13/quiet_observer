# Quiet Observer

Local computer vision app: capture frames from YouTube live streams, label them, train a YOLO model, and run continuous inference — all on your own computer.

See ARCHITECTURE.md for the technical architecture.

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
uv run uvicorn quiet_observer.main:app --host 127.0.0.1 --port 8000 --reload
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

## Potential future improvements

- Labelling helper that draws boxes around objects by comparing to previous frames
- Dark mode
- Make navigation more consistent
- Redesign monitor page
- Export of detection data 
- Export of trained models
- User-editable settings for model hyperparameters
- Fix confusing Clear all button logic on labelling page
- Add max samples per hour, or implement change detection frame selection: Implement ChangeDetectionStrategy
- Better dataset management: add train/val split control and “hard negatives”
- Model export: ONNX export for faster CPU inference if needed
- Smarter active learning: embedding-based novelty sampling, disagreement sampling (champion/challenger)
- Clean database.py by removing unneeded migrations.

### Training

- Labelling should allow setting frames as "not containing classes", then training should use those as well, ignoring only frames that have no labelling data attached.
- If the system crashes, it should afterwards show the training run as failed. Now it's forever in "running" state.
- Labelling UI should easily allow A) labelling images that have not been labelled yet, B) re-examining images that have been labelled already.

### Monitor page
- **Group detections by frame** — the current detection table has one row per detection; grouping by frame with bounding boxes drawn on a thumbnail would be more readable.
- **Parse `class_map_json` for display** — the deployed model's class map is currently shown as a raw JSON string; render it as a formatted class list.
- **Show review reasons as text** — review queue items show reason only as a tooltip on the `?` badge; surface `no_detection` / `low confidence (43%)` as readable text.
- **Show a "last processed" summary** — a header line with timestamp and detection count per tick makes it easier to confirm inference is actually running.
- **Confidence timeline or histogram** — aggregate detection stats over time per class.

### Data consistency
- **Training should use annotation existence, not the `is_labeled` flag** — `start_training` currently selects frames where `Frame.is_labeled == True`, but the project page already computes labeled counts from actual `Annotation` rows. Align `start_training` to use a subquery on `Annotation.frame_id` so both are consistent.

### Scalability
- **Paginate the detection list and review queue** — both are currently capped at fixed limits (50 and 10); add pagination or infinite scroll for projects with many frames.
