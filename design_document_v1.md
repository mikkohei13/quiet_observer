# Local Computer Vision from Video Design Doc (v1)

## 1) Purpose

Build a local app that turns a videos (YouTube live stream URL) into a continuously running object detector, with minimal setup:

Per-project: one stream URL, its dataset, its model versions, and its inference loop.

Support multiple projects in parallel (at least logically; run inference workers per project).

## 2) Constraints & principles

Simplicity first (one-person project). Favor simple solutions. Avoid premature optimization and over-engineering.

Runs fully locally on Mac M4 Pro.

Uses uv for Python dependency management (fast iteration)

Also Containers (OrbStack) for reproducible runs, but only if these bring clear value over added complexity

Frame capture: fixed interval only in v1 but design leaves a clear extension point for adding change detection later.

## 3) Non-goals (v1)

No multi-user auth/roles

No distributed training or cloud GPUs

No full labeling workflow management (review/consensus, etc.)

No real-time multi-camera orchestration at scale

## 4) Proposed architecture (minimal services)

Single Python app + background workers

One repo, one process for API/UI, plus one lightweight worker process for ingest + one for inference (or a single worker that does both). Training runs as a background job.

### Components:

Web App (UI + API)

Tech: FastAPI (API) + simple UI (either Streamlit embedded separately, or a small frontend served by FastAPI). Only minimal JavaScript.

Worker (background jobs)

Handles scheduled frame capture, training jobs, inference loops.

Tech: asyncio tasks or a small job queue (see below).

## 5) Key modules and responsibilities

### 5.1 Project Manager

Create/list projects

Validate YouTube URL

Store project configuration:

capture_interval_seconds

inference_interval_seconds

### 5.2 Stream Resolver + Frame Capture

Goal: reliably grab frames from YouTube live streams.

Resolve YouTube URL → streamable URL using e.g. yt-dlp

Capture frames using FFmpeg (most robust for streams)

Save frames to as files and index in DB

v1 implementation: FixedIntervalStrategy

v2: ChangeDetectionStrategy (drop-in replacement that decides whether to persist a candidate frame)

### 5.3 Labeling

UI shows frames; user draws boxes and assigns class

Allows adding and merging classes

Store annotations in a standard format:

internally: DB tables

export for training: YOLO format (or COCO)

Keep it simple: a lightweight bbox-drawing UI + server endpoints:

GET /projects/{id}/frames

POST /projects/{id}/annotations (upsert annotations for a frame)

### 5.4 Training Pipeline

Takes a snapshot of labeled data (dataset version)

Fine-tunes a base detector

Produces artifacts + metrics

Registers a model_version

Model choice (v1): Ultralytics YOLO (fast iteration), which works well for “birds at feeder”-type detection

On Apple Silicon, you can run training/inference via PyTorch MPS (Metal).

Artifacts to save:

- weights (best/last)

- class map

- training config

- metrics summary (mAP, loss curves)

- sample predictions on validation frames

### 5.5 Inference + Monitoring

Per project, the worker:

loads the currently deployed model_version

captures a frame (either from stored stream snapshot flow or direct)

runs inference at inference_interval_seconds

stores detections + (optionally) annotated preview images

### 5.6 Review Queue (basic “active learning”)

When inference is uncertain:

push frame into review_queue

user labels it later

create a new dataset version and retrain

Uncertainty heuristic (simple):

if max_confidence < threshold OR “no detection” for long time OR “new class requested”

## 6) Storage design (local-first)

### 6.1 Data store

SQLite is enough for v1 and keeps setup trivial.

Enables full local mode without running Postgres.

Can be swapped later (SQLAlchemy makes this easy).

### 6.2 File storage

Local filesystem:

data/projects/{project_id}/frames/{frame_id}.jpg

data/projects/{project_id}/runs/{run_id}/... (training logs, weights, exports)

data/projects/{project_id}/previews/... (optional annotated images)

This keeps everything inspectable and easy to back up.

## 7) Data model (minimal tables)

- projects

- frames

- classes

- annotations

- id

- dataset_versions

- dataset_version_frames (join table)

- training_runs

- model_versions

- deployments

- detections

- review_queue

Dataset snapshot approach: keep it simple: dataset_version is a list of frame IDs at creation time.

## 8) Core workflows

### 8.1 Create project

User submits name + YouTube URL + capture interval + inference interval

System resolves stream (sanity check)

Creates project and starts capture job

### 8.2 Capture frames (v1)

Every capture_interval_seconds:

grab one frame via FFmpeg from stream URL

store JPEG and create frames row

### 8.3 Label

UI lists recent frames + “unlabeled” filter

User draws boxes, assigns classes

Store in annotations

### 8.4 Train

User clicks “Train”

Create dataset_version snapshot: select frames with annotations

Start training_run

Fine-tune base model

Create model_version with metrics/artifacts

### 8.5 Deploy & monitor

User selects model_version → Deploy

Worker loads deployed model and begins inference loop

Detections appear in UI timeline; optional exports (CSV/JSON)

### 8.6 Improve (review queue)

Frames pushed to review_queue

User labels them

New training run → deploy updated model

## 9) Tech choices (Mac-friendly)

Note: you can change tech choices if needed, these are just suggestions.

Video / frames

yt-dlp to resolve live stream URL

ffmpeg to capture frames (subprocess call is fine)

ML

ultralytics (YOLO)

torch with MPS backend for Apple Silicon

albumentations (optional, later)

App

FastAPI + uvicorn

UI: start with Streamlit (fastest) or a minimal frontend

ORM: SQLAlchemy or SQLModel

Migrations: alembic (optional initially; can start without migrations)

Jobs

v1: asyncio background tasks (simple)

v1.5: RQ + Redis (cleaner job boundaries)

## 10) Local deployment modes

Mode 1: uv (recommended for speed)

uv manages dependencies and virtual env

Install system deps: ffmpeg, yt-dlp (can be via brew)

Extend with containers if needed.

## 11) Observability & debugging (keep lightweight)

Structured logs to file per project/job

Training run logs persisted under runs/{run_id}/

A simple “system status” page:

- running capture loops per project

- last frame captured time

- inference last run time

- queue of training jobs

## 12) Future ideas, exclude these from v1

Change detection frame selection:

Implement ChangeDetectionStrategy and plug into capture loop

Better dataset management:

add train/val split control and “hard negatives”

Model export:

ONNX export for faster CPU inference if needed

Smarter active learning:

embedding-based novelty sampling

disagreement sampling (champion/challenger)