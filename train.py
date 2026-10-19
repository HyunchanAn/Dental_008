import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def compute_sha256(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train leak-free YOLOv8m-seg for Dental_008 on RTX 5080")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.expanduser("~/.cache/dentex_dataset/yolo_format_v3/data.yaml"),
        help="Path to leak-free YOLO data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    base_pt = repo_root / "yolov8m-seg.pt"
    local_cache_root = Path(os.path.expanduser("~/.cache/dentex_dataset"))
    local_cache_root.mkdir(parents=True, exist_ok=True)
    local_base_pt = local_cache_root / "yolov8m-seg.pt"
    if not local_base_pt.exists() and base_pt.exists():
        shutil.copy2(base_pt, local_base_pt)
    if local_base_pt.exists():
        base_pt = local_base_pt

    local_cache_runs = local_cache_root / "runs" / "segment"
    local_cache_runs.mkdir(parents=True, exist_ok=True)
    run_name = "yolov8m_seg_leakfree_v4"

    print("============================================================")
    print("Dental_008 Leak-Free YOLOv8m-seg Training (RTX 5080)")
    print(f"Data YAML : {args.data}")
    print(f"Base Model: {base_pt}")
    print(f"Epochs    : {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}, Workers: {args.workers}")
    print("============================================================")

    model = YOLO(str(base_pt))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.epochs,
        project=str(local_cache_runs),
        name=run_name,
        device=args.device,
        workers=args.workers,
        amp=True,
        exist_ok=True,
    )

    best_pt = local_cache_runs / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected trained best.pt not found at {best_pt}")

    # Copy trained best.pt to Dental_008 weights and models directories
    weights_dir = repo_root / "weights"
    models_dir = repo_root / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    target_paths = [
        weights_dir / "best.pt",
        weights_dir / "yolov8m_best.pt",
        models_dir / "yolov8m_best.pt",
    ]
    for tp in target_paths:
        shutil.copy2(best_pt, tp)
        print(f"[train.py] Synced trained weights -> {tp}")

    best_sha256 = compute_sha256(best_pt)
    print(f"[train.py] Trained best.pt SHA-256: {best_sha256}")

    # Evaluate on leak-free validation split (127 images)
    print("\n[train.py] Running formal validation on leak-free Val (127 images)...")
    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )

    metrics_summary = {
        "weights_path": str(weights_dir / "best.pt"),
        "weights_sha256": best_sha256,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "train_images": 567,
        "dentex_train_images": 507,
        "edentulous_train_images": 60,
        "val_images": 127,
        "edentulous_holdout_images": 15,
        "box": {
            "precision": round(float(val_metrics.box.mp), 4),
            "recall": round(float(val_metrics.box.mr), 4),
            "map50": round(float(val_metrics.box.map50), 4),
            "map50_95": round(float(val_metrics.box.map), 4),
        },
        "mask": {
            "precision": round(float(val_metrics.seg.mp), 4),
            "recall": round(float(val_metrics.seg.mr), 4),
            "map50": round(float(val_metrics.seg.map50), 4),
            "map50_95": round(float(val_metrics.seg.map), 4),
        },
    }

    try:
        from scripts.test_on_edentulous import evaluate_holdout_edentulous
        ed_summary_025 = evaluate_holdout_edentulous(new_model_path=str(best_pt), conf=0.25)
        ed_summary_020 = evaluate_holdout_edentulous(new_model_path=str(best_pt), conf=0.20)
        metrics_summary["edentulous_holdout_conf025"] = ed_summary_025
        metrics_summary["edentulous_holdout_conf020"] = ed_summary_020
    except Exception as e:
        print(f"[train.py] Warning: Holdout edentulous evaluation error: {e}")

    local_metrics_path = Path(os.path.expanduser("~/.cache/dentex_dataset/dental_008_leak_free_val_metrics.json"))
    with open(local_metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_summary, mf, indent=2)

    dental_000_archive = repo_root.parent / "Dental_000" / "reports_archive"
    if dental_000_archive.exists():
        with open(dental_000_archive / "dental_008_leak_free_val_metrics.json", "w", encoding="utf-8") as mf:
            json.dump(metrics_summary, mf, indent=2)

    print("\n=== Leak-Free Val (127 Images) & Edentulous Holdout (15 Images) Quantitative Metrics ===")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
