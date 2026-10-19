import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from ultralytics import YOLO


def compute_sha256(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_holdout_edentulous(
    new_model_path: str | None = None,
    old_model_path: str | None = None,
    holdout_dir: str | None = None,
    train_dir: str | None = None,
    conf: float = 0.25,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    v3_root = Path(os.path.expanduser("~/.cache/dentex_dataset/yolo_format_v3"))

    holdout_path = Path(holdout_dir) if holdout_dir else (v3_root / "images" / "edentulous_val")
    train_path = Path(train_dir) if train_dir else (v3_root / "images" / "train")

    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout edentulous directory not found: {holdout_path}. Run scripts/prepare_v3_dataset.py first."
        )

    holdout_imgs = sorted(glob.glob(str(holdout_path / "*.jpg")))
    if not holdout_imgs:
        raise RuntimeError(f"No holdout images found in {holdout_path}")

    # Verify zero overlap against training images
    if train_path.exists():
        train_files = set(os.listdir(train_path))
        holdout_files = {os.path.basename(p) for p in holdout_imgs}
        overlap_names = train_files & holdout_files
        assert len(overlap_names) == 0, f"Leakage detected in holdout evaluation: {overlap_names}"

        train_ed_hashes = {
            compute_sha256(train_path / f)
            for f in train_files
            if f.lower().endswith(".jpg")
        }
        holdout_hashes = {compute_sha256(p) for p in holdout_imgs}
        assert len(train_ed_hashes & holdout_hashes) == 0, "SHA-256 overlap detected between train and holdout!"

    new_candidates = [
        new_model_path,
        str(repo_root / "weights" / "best.pt"),
        str(repo_root / "weights" / "yolov8m_best.pt"),
        str(repo_root / "models" / "yolov8m_best.pt"),
    ]
    resolved_new = next((p for p in new_candidates if p and os.path.exists(p)), None)
    if not resolved_new:
        raise FileNotFoundError(f"New model weights not found among candidates: {new_candidates}")

    old_candidates = [
        old_model_path,
        str(repo_root / "runs" / "segment" / "yolo_dentex" / "yolov8m_seg_run" / "weights" / "best.pt"),
    ]
    resolved_old = next((p for p in old_candidates if p and os.path.exists(p)), None)

    model_new = YOLO(resolved_new)
    model_old = YOLO(resolved_old) if resolved_old else None

    total_old_fp = 0
    total_new_fp = 0
    zero_fp_images_new = 0
    zero_fp_images_old = 0
    per_image = []

    print(f"Evaluating on {len(holdout_imgs)} independent holdout edentulous images (conf={conf}, overlap=0):")
    for img_path in holdout_imgs:
        fname = os.path.basename(img_path)
        res_new = model_new(img_path, conf=conf, verbose=False)[0]
        c_new = len(res_new.boxes) if res_new.boxes is not None else 0
        total_new_fp += c_new
        if c_new == 0:
            zero_fp_images_new += 1

        c_old = None
        if model_old is not None:
            res_old = model_old(img_path, conf=conf, verbose=False)[0]
            c_old = len(res_old.boxes) if res_old.boxes is not None else 0
            total_old_fp += c_old
            if c_old == 0:
                zero_fp_images_old += 1
            print(f"  {fname} -> Baseline(v2) FP: {c_old}, Retrained(v4-Holdout) FP: {c_new}")
        else:
            print(f"  {fname} -> Retrained(v4-Holdout) FP: {c_new}")

        per_image.append({"filename": fname, "baseline_fp": c_old, "retrained_fp": c_new})

    n_imgs = len(holdout_imgs)
    fp_per_img_new = total_new_fp / n_imgs
    fp_per_img_old = (total_old_fp / n_imgs) if model_old is not None else None
    suppression_rate = (
        ((total_old_fp - total_new_fp) / total_old_fp * 100.0)
        if (model_old is not None and total_old_fp > 0)
        else (100.0 if total_new_fp == 0 else 0.0)
    )

    summary = {
        "holdout_image_count": n_imgs,
        "conf_threshold": conf,
        "new_model_path": str(resolved_new),
        "new_model_sha256": compute_sha256(resolved_new),
        "baseline_model_path": str(resolved_old) if resolved_old else None,
        "baseline_total_fp": total_old_fp if model_old is not None else None,
        "baseline_fp_per_image": round(fp_per_img_old, 4) if fp_per_img_old is not None else None,
        "baseline_zero_fp_images": zero_fp_images_old if model_old is not None else None,
        "retrained_total_fp": total_new_fp,
        "retrained_fp_per_image": round(fp_per_img_new, 4),
        "retrained_zero_fp_images": zero_fp_images_new,
        "retrained_zero_fp_rate_pct": round(zero_fp_images_new / n_imgs * 100.0, 2),
        "fp_suppression_rate_pct": round(suppression_rate, 2),
        "per_image": per_image,
    }

    print("\n=== Independent Holdout Edentulous Benchmark Summary ===")
    print(f"Holdout Images (Unseen in Training): {n_imgs}")
    if model_old is not None:
        print(f"Baseline (v2, Pre-HardNegative) Total FP: {total_old_fp} ({fp_per_img_old:.2f} FP/img)")
    print(f"Retrained (v4, 60/15 Split) Total FP: {total_new_fp} ({fp_per_img_new:.2f} FP/img)")
    print(f"Zero-FP Image Rate: {zero_fp_images_new}/{n_imgs} ({summary['retrained_zero_fp_rate_pct']:.1f}%)")
    print(f"FP Suppression Rate vs Baseline: {summary['fp_suppression_rate_pct']:.2f}%")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate false positives on independent holdout edentulous images")
    parser.add_argument("--model-new", type=str, default=None, help="Path to newly trained YOLOv8m-seg weights")
    parser.add_argument("--model-old", type=str, default=None, help="Path to baseline v2 weights")
    parser.add_argument("--holdout-dir", type=str, default=None, help="Path to independent 15 holdout edentulous images")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to training images for disjointness check")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    evaluate_holdout_edentulous(
        new_model_path=args.model_new,
        old_model_path=args.model_old,
        holdout_dir=args.holdout_dir,
        train_dir=args.train_dir,
        conf=args.conf,
    )
