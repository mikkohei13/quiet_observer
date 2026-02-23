# Architecture

FastAPI server-side rendered app for continuous object detection on YouTube live streams. Captures frames via yt-dlp/ffmpeg, runs YOLO inference, and provides a labeling UI to fine-tune custom models.

**Terminology**: "capture" means the generic act of grabbing a frame from a video stream (yt-dlp + ffmpeg). "Sample" means storing a captured frame for the labeling/training pipeline. The sampling worker and the inference worker both capture frames, but decide independently which ones to keep as samples.

## Stack

- **Server**: FastAPI + Uvicorn, Jinja2 templates, no frontend framework
- **Database**: SQLite via SQLAlchemy ORM (`data/quiet_observer.db`), `create_all()` on startup + lightweight ALTER TABLE migrations in `init_db()`
- **ML**: Ultralytics YOLO (`yolo11n.pt` base), fine-tuned per project
- **Video**: yt-dlp resolves stream URL, ffmpeg grabs single JPEG frames
- **Workers**: asyncio background tasks managed by a singleton `WorkerManager`

## Directory layout

```
src/quiet_observer/
├── main.py              # App factory, lifespan (init_db / stop workers), router mounting
├── config.py            # Paths (DATA_DIR, TEMPLATES_DIR, STATIC_DIR), DATABASE_URL, constants
├── database.py          # Engine, SessionLocal, get_db() dependency, init_db()
├── models.py            # 12 SQLAlchemy models, no relationship() declarations
├── routers/
│   ├── projects.py      # Project CRUD, sampling/inference start/stop, live inference JSON APIs
│   ├── frames.py        # GET /frames/{id}/image — serves JPEGs from disk
│   ├── annotations.py   # Labeling UI, annotation save/replace, class CRUD
│   ├── training.py      # Training dashboard, start training, deploy model
│   └── monitoring.py    # Monitor dashboard, system status page
├── workers/
│   ├── manager.py       # WorkerManager singleton — start/stop/track asyncio tasks per project
│   ├── capture.py       # capture utilities (yt-dlp, ffmpeg) + sample_loop()
│   └── inference.py     # inference_loop(): capture frame → run YOLO → Detection rows
├── ml/
│   └── trainer.py       # Dataset export (YOLO format), model.train(), ModelVersion creation
└── templates/           # 12 Jinja2 templates (base, projects, detail, frames_browse, label, train, monitor, etc.)

static/
├── style.css            # Full app stylesheet
└── label.js             # Canvas-based bounding box annotation tool

data/                    # Runtime — DB, sampled/inferred frames, training runs (gitignored)
```

## Data model

```
Project ──< Frame ──< Annotation >── Class
   │           │
   │           ├──< Detection (from inference)
   │           │
   │           └──< ReviewQueue (sampled frames for labeling)
   │
   ├──< DatasetVersion ──< DatasetVersionFrame >── Frame
   │
   ├──< TrainingRun ──< ModelVersion ──< Deployment
   │
   └──< InferenceSession
```

No ORM relationships are declared. All joins are explicit `db.query().filter()` calls. Bounding boxes (Annotation, Detection) use normalized YOLO format: `x_center, y_center, width, height` in 0.0–1.0 range.

Key columns worth knowing:
- `Project.sampling_active` / `inference_active` — persisted intended state (not used for auto-restart)
- `Project.last_inference_at` — updated per frame during inference
- `Project.last_inferred_frame_id` — tracks the latest frame processed by inference (including zero-detection frames); used by `/inference/latest` and `/inference/recent` endpoints; cleared on inference start so the UI resets
- `Detection.detected_at` — explicit `datetime.utcnow()`
- `Project.auto_sample_interval_seconds` / `low_confidence_threshold` / `high_confidence_threshold` — per-project inference sampling settings (defaults 600 / 0.3 / 0.7), editable via the project edit form
- `ReviewQueue.reason` — `"auto_sample"` or `"uncertain_confidence:0.45"` etc.
- `InferenceSession.stopped_at` — `None` means running or crashed; orphans closed on next start
- `Frame.file_path` — relative to `DATA_DIR`
- `Frame.source` — `"sampler"` (from sampling worker) or `"inference"` (from inference worker)
- `Frame.label_status` — `"unlabeled"` (default), `"annotated"` (has bounding boxes), or `"negative"` (explicitly marked as containing no objects; YOLO background sample). Both `"annotated"` and `"negative"` count as labeled for project stats and training. Migrated from the earlier `is_labeled` boolean via `init_db()`.

## Worker system

`WorkerManager` (singleton at module level) holds `dict[int, asyncio.Task]` for sampling and inference tasks, keyed by project_id.

**Sampling loop** (`capture.py: sample_loop()`) — samples frames at a fixed interval for labeling and training:
1. Resolve YouTube stream URL via `yt-dlp --get-url`
2. Capture one frame via `ffmpeg -i {url} -frames:v 1`
3. Save to `data/projects/{id}/frames/{timestamp}.jpg`
4. Insert `Frame` row (`source="sampler"`), sleep `sample_interval_seconds`

**Inference loop** (`inference.py`) — runs detections on a live stream and selectively samples frames:
1. Load deployed YOLO model (cached, reloaded on version change)
2. Capture a frame from the YouTube stream (same yt-dlp/ffmpeg mechanism; stream URL cached, re-resolved on failure)
3. Save frame to disk, insert `Frame` row (`source="inference"`)
4. Run model via `run_in_executor` (thread pool) with `conf=YOLO_INFERENCE_CONF` (default 0.1)
5. Write `Detection` rows, update `project.last_inferred_frame_id`
6. Sample frame for labeling (add to `ReviewQueue`) if either condition is met:
   - Any detection confidence in [low_threshold, high_threshold] — the uncertain range worth human review (detections below low_threshold are treated as noise)
   - Time since last sample ≥ `auto_sample_interval_seconds`
   - Thresholds are per-project (configurable in project edit form), with config.py defaults as fallback
7. Sleep `inference_interval_seconds`

Sampling and inference are independent — either can run without the other. Both create `Frame` rows from the YouTube stream. The `Frame.source` column distinguishes their origin.

The labeling UI prioritizes ReviewQueue items first, then falls back to unlabeled `source="sampler"` frames. Inference frames only reach the labeling pipeline when explicitly sampled via the ReviewQueue.

Workers are fire-and-forget asyncio tasks. Stopped via `task.cancel()`. All stopped on app shutdown via lifespan.

## Training pipeline

Triggered from `/projects/{id}/train/start`:
1. Snapshot: copies current labeled frame IDs (annotated + negative) into a `DatasetVersion`
2. Export: writes YOLO-format dataset (images + label .txt files, randomized 80/20 split). Negative frames get empty `.txt` label files so YOLO treats them as background samples.
3. Train: `YOLO(base_model).train(...)` in thread pool, logs to `data/projects/{id}/runs/{run_id}/train.log`
4. On success: create `ModelVersion` with weights path + metrics JSON
5. User deploys via POST to `/model_versions/{id}/deploy` (deactivates previous)

Training is a fire-and-forget `asyncio.create_task` — no reference stored, no cancellation support.

Fine-tuning defaults (in `config_json`): `epochs=100`, `freeze=10` (backbone layers frozen to prevent catastrophic forgetting), `lr0=0.001`, `patience=20` (early stopping). YOLO `verbose=True` so per-epoch output is captured in the log.

**Training log UI** (`/training_runs/{id}/log`): shows run metadata (config, timing, status), YOLO-generated plot images (training curves, confusion matrices, F1/PR curves, val predictions), per-epoch metrics table parsed from `results.csv`, and the raw log. Auto-refreshes while status is `running`. Plot images and other run files are served via `/training_runs/{id}/files/{path}`.

## Live inference UI

`project_detail.html` contains inline JS that:
1. Polls `GET /projects/{id}/inference/latest` at `inference_interval_seconds` intervals
2. Draws frame image + detection bounding boxes on an HTML5 canvas
3. Only redraws when `frame.id` changes (dedup via `currentFrameId`)
4. Resets to empty state when `inference_running` is false or no frame available
5. On new frame: also fetches `GET /projects/{id}/inference/recent` and rebuilds the recent results grid

The `class_color_map` (from DB `Class.color`) is passed to JS as a JSON object for consistent pill/box coloring.

## Route map (25 routes)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Project list |
| GET/POST | `/projects/new`, `/projects` | Create project |
| GET | `/projects/{id}` | Project dashboard |
| GET | `/projects/{id}/frames?filter=` | Browse frames by category (all, annotated, negative, unlabeled_samples, unlabeled_inference) |
| GET/POST | `/projects/{id}/edit` | Edit project |
| POST | `/projects/{id}/sampling/start\|stop` | Control sampling worker |
| POST | `/projects/{id}/inference/start\|stop` | Control inference worker |
| GET | `/projects/{id}/inference/latest` | JSON: latest inferred frame + detections |
| GET | `/projects/{id}/inference/recent` | JSON: last 10 inferred frames |
| GET | `/frames/{id}/image` | Serve frame JPEG |
| GET | `/projects/{id}/label[/{frame_id}]` | Annotation UI |
| POST | `/projects/{id}/frames/{frame_id}/annotations` | Save annotations (JSON) |
| POST | `/projects/{id}/frames/{frame_id}/mark_negative` | Mark frame as negative (no objects) |
| POST | `/projects/{id}/classes` | Create class |
| POST | `/classes/{id}/rename\|delete` | Rename/delete class |
| GET | `/projects/{id}/train` | Training dashboard |
| POST | `/projects/{id}/train/start` | Start training run |
| POST | `/model_versions/{id}/deploy` | Deploy model version |
| GET | `/training_runs/{id}/log` | View training log + plots + metrics |
| GET | `/training_runs/{id}/files/{path}` | Serve training run files (plots, etc.) |
| GET | `/projects/{id}/monitor` | Monitoring dashboard |
| GET | `/status` | System-wide worker status |

## Notable patterns

- **No middleware, auth, or CORS** — designed for local/trusted use only
- **No ORM relationships** — manual query joins everywhere, batch-loads to avoid N+1
- **Worker ↔ router communication** — import `worker_manager` singleton directly; in-memory status checks
- **Session management** — each request gets a fresh `SessionLocal()` via `get_db()`. Workers create their own sessions per loop iteration.
- **SQLite concurrency** — `check_same_thread=False`, default journal mode (not WAL). Workers commit and close sessions before sleeping.
