import argparse
import glob
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
import yaml


def compute_sha256(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_v3_dataset(
    src_dir: str | None = None,
    dst_dir: str | None = None,
    edentulous_dir: str | None = None,
    seed: int = 42,
    train_edentulous_count: int = 60,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = Path(src_dir or os.path.expanduser("~/.cache/dentex_dataset/yolo_format"))
    dst_path = Path(dst_dir or os.path.expanduser("~/.cache/dentex_dataset/yolo_format_v3"))
    ed_path = Path(edentulous_dir) if edentulous_dir else (repo_root / "edentulous_images")

    if not src_path.exists():
        raise FileNotFoundError(f"Source leak-free YOLO dataset not found: {src_path}. Run prepare_yolo_data.py first.")
    if not ed_path.exists():
        raise FileNotFoundError(f"Edentulous images directory not found: {ed_path}")

    if dst_path.exists():
        shutil.rmtree(dst_path)

    for split in ("train", "val", "edentulous_val"):
        (dst_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("[prepare_v3_dataset] Copying leak-free DENTEX train (507) and val (127) splits...")
    dentex_counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        src_imgs = sorted(glob.glob(str(src_path / "images" / split / "*.*")))
        for src_img in src_imgs:
            filename = os.path.basename(src_img)
            if "hardneg_" in filename:
                continue
            shutil.copy2(src_img, dst_path / "images" / split / filename)
            basename = os.path.splitext(filename)[0]
            src_label = src_path / "labels" / split / f"{basename}.txt"
            dst_label = dst_path / "labels" / split / f"{basename}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            dentex_counts[split] += 1

    # Verify DENTEX train/val disjointness
    d_train = set(os.listdir(dst_path / "images" / "train"))
    d_val = set(os.listdir(dst_path / "images" / "val"))
    assert len(d_train & d_val) == 0, "CRITICAL: Overlap between DENTEX train and val!"

    # Deterministically split 75 edentulous images into 60 train (Hard Negative) and 15 holdout val
    all_ed_imgs = sorted(glob.glob(str(ed_path / "*.jpg")))
    if len(all_ed_imgs) < train_edentulous_count + 1:
        raise RuntimeError(f"Expected at least {train_edentulous_count + 1} edentulous images, found {len(all_ed_imgs)}")

    rng = random.Random(seed)
    shuffled_ed = list(all_ed_imgs)
    rng.shuffle(shuffled_ed)

    ed_train_imgs = sorted(shuffled_ed[:train_edentulous_count])
    ed_val_imgs = sorted(shuffled_ed[train_edentulous_count:])

    ed_train_names = {os.path.basename(p) for p in ed_train_imgs}
    ed_val_names = {os.path.basename(p) for p in ed_val_imgs}
    assert len(ed_train_names & ed_val_names) == 0, "CRITICAL: Edentulous train and holdout val filenames overlap!"

    ed_train_hashes = {compute_sha256(p) for p in ed_train_imgs}
    ed_val_hashes = {compute_sha256(p) for p in ed_val_imgs}
    assert len(ed_train_hashes & ed_val_hashes) == 0, "CRITICAL: Edentulous train and holdout val SHA-256 hashes overlap!"

    print(
        f"[prepare_v3_dataset] Merging {len(ed_train_imgs)} edentulous images into train "
        f"and isolating {len(ed_val_imgs)} into holdout edentulous_val (0 overlap)..."
    )

    for img_path in ed_train_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, dst_path / "images" / "train" / filename)
        basename = os.path.splitext(filename)[0]
        with open(dst_path / "labels" / "train" / f"{basename}.txt", "w", encoding="utf-8") as f:
            pass  # Empty background label (0 objects)

    for img_path in ed_val_imgs:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, dst_path / "images" / "edentulous_val" / filename)
        basename = os.path.splitext(filename)[0]
        with open(dst_path / "labels" / "edentulous_val" / f"{basename}.txt", "w", encoding="utf-8") as f:
            pass  # Empty background label (0 objects)

    manifest_data = {
        "seed": seed,
        "dentex_train_count": dentex_counts["train"],
        "dentex_val_count": dentex_counts["val"],
        "edentulous_train_count": len(ed_train_imgs),
        "edentulous_holdout_val_count": len(ed_val_imgs),
        "total_train_count": dentex_counts["train"] + len(ed_train_imgs),
        "train_val_filename_overlap": 0,
        "edentulous_train_val_sha256_overlap": 0,
        "edentulous_train_files": sorted(ed_train_names),
        "edentulous_holdout_val_files": sorted(ed_val_names),
    }
    manifest_out = dst_path / "edentulous_split_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    data_yaml = {
        "path": dst_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Tooth"},
    }
    with open(dst_path / "data.yaml", "w", encoding="utf-8") as yf:
        yaml.dump(data_yaml, yf, default_flow_style=False, sort_keys=False)

    print(
        f"[prepare_v3_dataset] Complete! Train={manifest_data['total_train_count']} "
        f"(DENTEX {dentex_counts['train']} + Edentulous {len(ed_train_imgs)}), "
        f"Val={dentex_counts['val']}, Edentulous Holdout={len(ed_val_imgs)}"
    )
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare leak-free v3 dataset with edentulous holdout split")
    parser.add_argument("--src-dir", type=str, default=None)
    parser.add_argument("--dst-dir", type=str, default=None)
    parser.add_argument("--edentulous-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-edentulous-count", type=int, default=60)
    args = parser.parse_args()
    prepare_v3_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        edentulous_dir=args.edentulous_dir,
        seed=args.seed,
        train_edentulous_count=args.train_edentulous_count,
    )
