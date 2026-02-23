# Pretrained YOLO Mode (Simple Implementation Idea)

## Goal

Add a second project mode that uses YOLO’s pretrained model directly (no labeling/training), while keeping the current custom-training workflow unchanged.

- **Current mode:** custom project (label -> train -> deploy -> infer)
- **New mode:** pretrained project (select classes -> infer)

This lets users quickly do things like “detect cars” without collecting labels.

---

## Why this fits well

The app already has everything needed for live inference:
- stream capture
- YOLO model loading
- detection storage (`Detection`)
- live UI + monitor UI

So this is mostly a “new setup path”, not a rewrite.

---

## Minimal data model changes

### 1) Add project mode

Add a field to `Project`:

- `mode`: `"custom"` or `"pretrained"` (default `"custom"`)

### 2) Allow model versions without training runs

For pretrained projects, there is no training run.  
So `ModelVersion.training_run_id` should be nullable.

### 3) Store selected classes

Simplest path: reuse existing `Class` rows.
- On pretrained project creation, create `Class` rows from selected COCO classes (e.g. `car`, `truck`)
- Store COCO index mapping in `ModelVersion.class_map_json` (or another small metadata field)

This keeps UI coloring and detection display consistent with current logic.

---

## Simple backend flow

## Project creation (`mode=pretrained`)

When user creates a pretrained project:

1. Save `Project(mode="pretrained")`
2. Create `Class` rows for selected COCO classes
3. Create a synthetic `ModelVersion` pointing to base weights (e.g. `yolo11n.pt`)
4. Auto-create active `Deployment` for that model version

Result: inference can start immediately using existing deployment flow.

---

## Inference changes (small)

Current inference loop already loads deployed model weights and runs YOLO.
For pretrained projects:

- model weights path resolves to base YOLO weights (`yolo11n.pt`)
- pass selected class indices to YOLO (`classes=[...]`) so only requested classes are detected

Conceptually:

- custom mode -> deployed fine-tuned weights
- pretrained mode -> deployed base weights + class filter

Everything else (frame capture, DB writes, live polling) stays the same.

---

## UI changes (minimal)

## New project form

Add:
- mode selector: **Custom training** / **Pretrained**
- if pretrained: show COCO class multi-select
- if custom: keep current behavior

## Project detail page

If `project.mode == "pretrained"`:
- hide/disable labeling and training actions
- show active pretrained class list
- keep inference controls as normal

## Training page

If pretrained:
- show message like: “This project uses a pretrained model. Training is not required.”

Monitor/live inference pages should mostly work unchanged.

---

## Keep scope intentionally small

For first version, avoid extra complexity:
- no new inference architecture
- no big refactor
- no class taxonomy changes
- no migration of existing projects beyond defaults

Just add a clean “pretrained path” on top of current system.

---

## Example user journey (pretrained)

1. Create project -> choose **Pretrained**
2. Select classes: `car` (optionally `truck`, `bus`)
3. Click create
4. Start inference
5. See detections immediately

No labeling or training required.

---

## Optional future enhancement

Allow upgrading a pretrained project into custom mode:
- keep collected frames
- start labeling selected frames
- train specialized model later
- deploy fine-tuned model when ready