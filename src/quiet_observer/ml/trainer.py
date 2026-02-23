"""YOLO training pipeline."""
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, YOLO_BASE_MODEL
from ..database import SessionLocal
from sqlalchemy import func

from ..models import (
    Annotation, Class, DatasetVersionFrame,
    Frame, ModelVersion, TrainingRun,
)

logger = logging.getLogger(__name__)


def export_yolo_dataset(
    run: TrainingRun,
    db,
    dataset_dir: Path,
) -> dict:
    """Export labeled frames and annotations to YOLO format. Returns class map."""
    frame_ids = [
        dvf.frame_id
        for dvf in db.query(DatasetVersionFrame)
        .filter(DatasetVersionFrame.dataset_version_id == run.dataset_version_id)
        .all()
    ]

    classes = db.query(Class).filter(Class.project_id == run.project_id).all()
    class_map = {cls.id: idx for idx, cls in enumerate(classes)}
    class_names = {idx: cls.name for idx, cls in enumerate(classes)}

    # Shuffle before splitting so train/val aren't biased by sampling order
    random.seed(42)
    random.shuffle(frame_ids)

    # Split 80/20 train/val
    split_idx = max(1, int(len(frame_ids) * 0.8))
    train_ids = frame_ids[:split_idx]
    val_ids = frame_ids[split_idx:] if len(frame_ids) > 1 else frame_ids

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for frame_id in ids:
            frame = db.query(Frame).filter(Frame.id == frame_id).first()
            if not frame:
                continue

            src = DATA_DIR / frame.file_path
            if not src.exists():
                continue

            dst_img = img_dir / f"{frame_id}.jpg"
            shutil.copy2(src, dst_img)

            lbl_path = lbl_dir / f"{frame_id}.txt"
            if frame.label_status == "negative":
                # Empty label file = YOLO background/negative sample
                lbl_path.touch()
            else:
                annotations = db.query(Annotation).filter(Annotation.frame_id == frame_id).all()
                with open(lbl_path, "w") as f:
                    for ann in annotations:
                        cls_idx = class_map.get(ann.class_id)
                        if cls_idx is None:
                            continue
                        f.write(f"{cls_idx} {ann.x:.6f} {ann.y:.6f} {ann.width:.6f} {ann.height:.6f}\n")

    # Write dataset.yaml
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val
nc: {len(classes)}
names: {[cls.name for cls in classes]}
"""
    (dataset_dir / "dataset.yaml").write_text(yaml_content)

    return class_names


async def run_training(run_id: int) -> None:
    """Run YOLO fine-tuning for a training run. Called as asyncio background task."""
    import asyncio

    db = SessionLocal()
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        db.close()
        return

    run.status = "running"
    db.commit()

    run_dir = DATA_DIR / f"projects/{run.project_id}/runs/{run.id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    run.log_path = str(log_path)
    db.commit()

    log_file = open(log_path, "w")

    def log(msg: str):
        logger.info(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log(f"Training run {run_id} started at {datetime.utcnow()}")
        config = json.loads(run.config_json or "{}")
        epochs = config.get("epochs", 100)
        imgsz = config.get("imgsz", 640)
        freeze = config.get("freeze", 10)
        lr0 = config.get("lr0", 0.001)
        patience = config.get("patience", 20)

        dataset_dir = run_dir / "dataset"
        log("Exporting YOLO dataset...")
        class_names = export_yolo_dataset(run, db, dataset_dir)
        log(f"Dataset exported: {class_names}")

        # Log class distribution in the training dataset
        frame_ids = [
            dvf.frame_id
            for dvf in db.query(DatasetVersionFrame)
            .filter(DatasetVersionFrame.dataset_version_id == run.dataset_version_id)
            .all()
        ]
        class_counts = (
            db.query(Class.name, func.count(Annotation.id).label("cnt"))
            .join(Annotation, Annotation.class_id == Class.id)
            .filter(Annotation.frame_id.in_(frame_ids))
            .group_by(Class.id, Class.name)
            .order_by(func.count(Annotation.id).desc())
            .all()
        )
        negative_count = (
            db.query(Frame)
            .filter(Frame.id.in_(frame_ids), Frame.label_status == "negative")
            .count()
        )
        log(f"Training on {len(frame_ids)} frame(s) ({negative_count} negative) — class distribution:")
        for cls_name, cnt in class_counts:
            log(f"  {cls_name}: {cnt} annotation(s)")
        if not class_counts and negative_count == 0:
            log("  (no annotations found)")

        yaml_path = dataset_dir / "dataset.yaml"

        # Run YOLO training in a thread pool (CPU/GPU bound)
        loop = asyncio.get_event_loop()

        def _train():
            from ultralytics import YOLO
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            log(f"Training on device: {device}")
            log(f"Hyperparameters: epochs={epochs}, imgsz={imgsz}, freeze={freeze}, lr0={lr0}, patience={patience}")
            log(f"Base model: {YOLO_BASE_MODEL}")

            model = YOLO(YOLO_BASE_MODEL)
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                project=str(run_dir),
                name="yolo",
                device=device,
                verbose=True,
                plots=True,
                freeze=freeze,
                lr0=lr0,
                patience=patience,
            )
            return results

        results = await loop.run_in_executor(None, _train)

        # Find best weights
        weights_dir = run_dir / "yolo" / "weights"
        best_weights = weights_dir / "best.pt"
        if not best_weights.exists():
            best_weights = weights_dir / "last.pt"

        if not best_weights.exists():
            raise FileNotFoundError(f"No weights found in {weights_dir}")

        # Collect metrics
        metrics = {}
        try:
            results_csv = run_dir / "yolo" / "results.csv"
            if results_csv.exists():
                import csv
                with open(results_csv) as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        last = rows[-1]
                        metrics = {k.strip(): v.strip() for k, v in last.items()}
                log(f"Final epoch metrics:")
                for k, v in metrics.items():
                    log(f"  {k}: {v}")
                log(f"Total epochs completed: {len(rows)}")
        except Exception as e:
            log(f"Could not parse metrics: {e}")

        # Register model version
        mv = ModelVersion(
            project_id=run.project_id,
            training_run_id=run.id,
            weights_path=str(best_weights),
            metrics_json=json.dumps(metrics),
            class_map_json=json.dumps(class_names),
        )
        db.add(mv)

        run.status = "done"
        run.finished_at = datetime.utcnow()
        db.commit()
        log(f"Training completed in {(run.finished_at - run.started_at).total_seconds():.1f}s. Model version saved.")
        log(f"Weights: {best_weights}")
        log(f"Class map: {class_names}")

    except Exception as e:
        logger.exception("Training failed for run %d: %s", run_id, e)
        run.status = "failed"
        run.error_message = str(e)
        run.finished_at = datetime.utcnow()
        db.commit()
        log(f"Training FAILED: {e}")
    finally:
        log_file.close()
        db.close()
